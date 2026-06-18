"""Reusable live ability detector.

Wraps the perf-tuned screen-capture + model loop from :mod:`src.infer.live`
into a class that reports detections through an ``on_event`` callback instead
of printing. Used by:

- the web server (:mod:`src.app.server`) to auto-start cooldown timers, and
- the ``src.infer.live`` CLI, which passes print callbacks (so the tuned loop
  lives in exactly one place).

The heavy building blocks (``ScreenGrabber``, ``StreamDetector``,
``_prepare_live_clip``, ``resolve_region``, ``_live_cfg``) are imported from
``src.infer.live`` to avoid duplicating the capture/inference logic.
"""
from __future__ import annotations

import os

# x3d uses avg_pool3d, unimplemented on MPS; allow CPU fallback. Must precede torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import threading
import time
from collections import Counter, deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.common.config import Config
from src.common.torch_utils import pick_device
from src.dataset.clip_dataset import ClipTransform
from src.infer.live import (
    ScreenGrabber,
    StreamDetector,
    _live_cfg,
    _prepare_live_clip,
    resolve_region,
)
from src.infer.recognize import _localizer_from_ckpt, load_model
from src.localize.level_reader import LevelReader

# Callback signatures.
EventCallback = Callable[[str, float, float, str], None]  # ability, score, time, source
StatusCallback = Callable[[Dict], None]                  # arbitrary status dict


class _LevelStabilizer:
    """Require repeated reads before emitting a level change."""

    def __init__(self, window: int = 5, min_agree: int = 2) -> None:
        self._window = window
        self._min_agree = min_agree
        self._history: Deque[int] = deque(maxlen=window)
        self._last: Optional[int] = None

    def push(self, level: int, confidence: float, min_conf: float = 0.5) -> Optional[int]:
        if confidence < min_conf:
            return self._last
        self._history.append(int(level))
        if len(self._history) < self._min_agree:
            return self._last
        level, count = Counter(self._history).most_common(1)[0]
        if count >= self._min_agree:
            self._last = level
        return self._last


class LiveDetector:
    """Loads a checkpoint and runs the live capture/inference loop.

    Parameters mirror the ``src.infer.live`` CLI. Detections are delivered via
    ``on_event(ability, score, clip_time)``; periodic telemetry (capture fps,
    timing, top-1 readout) via ``on_status(dict)``.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint: str,
        device_str: str = "auto",
        monitor: int = 1,
        region: Optional[str] = None,
        stride_sec: Optional[float] = None,
        on_event: Optional[EventCallback] = None,
        on_status: Optional[StatusCallback] = None,
        status_interval_sec: float = 0.5,
    ) -> None:
        self.on_event = on_event
        self.on_status = on_status
        self.status_interval_sec = status_interval_sec

        cfg = Config.load(config_path)
        self.champion = cfg.champion
        icfg = cfg.section("infer")
        lcfg = _live_cfg(icfg)
        self.device = pick_device(device_str)

        track_list = [str(x) for x in (icfg.get("track") or [])]
        emit_list = [str(x) for x in (icfg.get("emit") or [])]
        emit_classes: List[str] = []
        seen: set[str] = set()
        for key in track_list + emit_list:
            if key not in seen:
                seen.add(key)
                emit_classes.append(key)
        self._emit_classes = emit_classes or None

        timers_cfg = cfg.section("timers")
        r_bar_cfg = timers_cfg.get("r_bar") or {}
        self._r_bar_ability = str(r_bar_cfg.get("ability", "R"))

        self.model, self.ckpt = load_model(checkpoint, self.device)
        self.classes: List[str] = self.ckpt["classes"]
        self.num_frames = int(self.ckpt["num_frames"])
        self.sample_fps = float(self.ckpt["sample_fps"])
        self.crop_size = int(self.ckpt["crop_size"])
        self.frame_mode = str(self.ckpt.get("frame_mode", "center_crop"))
        self.hud_mask = self.ckpt.get("hud_mask") or []
        self.localizer = _localizer_from_ckpt(self.ckpt)
        loc_cfg = cfg.section("localize")
        lr_cfg = loc_cfg.get("level_read") or {}
        self.level_reader = (
            LevelReader.from_config(lr_cfg, base_dir=".")
            if bool(lr_cfg.get("enabled"))
            else None
        )
        self._level_stabilizer = _LevelStabilizer()
        self.clip_dur = self.num_frames / self.sample_fps
        self.transform = ClipTransform(
            self.crop_size, self.ckpt["mean"], self.ckpt["std"],
            train=False, frame_mode=self.frame_mode,
        )

        self.stride = float(
            stride_sec if stride_sec is not None else lcfg.get("stride_sec", 0.25))
        self.detector = StreamDetector(
            self.classes,
            thresholds=lcfg.get("thresholds", {}) or {},
            default_threshold=float(lcfg.get("default_threshold", 0.6)),
            nms_window_sec=float(lcfg.get("nms_window_sec", 1.0)),
            peak_only=bool(lcfg.get("peak_only", False)),
            min_margin=float(lcfg.get("min_margin", 0.0)),
            track=self._emit_classes,
        )
        self.loc_cache = {"sec": float(lcfg.get("localize_cache_sec", 0.0))}

        from src.timers import load_r_bar

        self.r_bar = load_r_bar(cfg, base_dir=".")

        self.region = resolve_region(monitor, region)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> None:
        """Run the loop in a background daemon thread (for the server)."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- main loop ---------------------------------------------------------- #
    def run_loop(self, preview_cb: Optional[Callable[[ScreenGrabber], bool]] = None) -> None:
        """Blocking capture/inference loop.

        ``preview_cb`` (CLI only) is called each iteration with the grabber and
        should return ``False`` to stop (e.g. the preview window was closed).
        """
        grabber = ScreenGrabber(
            self.region, self.num_frames, self.sample_fps, self.crop_size,
            hud_mask=self.hud_mask, frame_mode=self.frame_mode,
            preview=preview_cb is not None,
        )
        grabber.start()
        next_infer = time.monotonic()
        last_status = 0.0
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if preview_cb is not None and not preview_cb(grabber):
                    break
                if now < next_infer:
                    time.sleep(min(0.005, next_infer - now))
                    continue
                next_infer += self.stride

                frames, center_t = grabber.snapshot()
                if frames is None:
                    next_infer = time.monotonic() + self.stride  # still buffering
                    continue

                clip = np.stack(frames, axis=0)  # (T, H, W, 3) uint8 full-res
                t0 = time.monotonic()
                model_clip, loc_ms = _prepare_live_clip(
                    clip, self.ckpt, self.localizer, self.loc_cache)
                if model_clip is None:
                    if self.on_status:
                        self.on_status({"localize_miss": True, "loc_ms": loc_ms})
                    continue

                x = self.transform(model_clip).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(x)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                model_ms = (time.monotonic() - t0 - loc_ms / 1000.0) * 1000.0

                for e in self.detector.push(center_t, probs):
                    if self.on_event:
                        self.on_event(
                            e["ability"], float(e["score"]), float(e["time"]), "vfx",
                        )

                level_payload = {}
                det = self.loc_cache.get("det")
                if det is not None and self.r_bar is not None:
                    center = clip[len(clip) // 2]
                    r_score = self.r_bar.update(center, det.bar, now)
                    if r_score is not None and self.on_event:
                        self.on_event(
                            self._r_bar_ability, float(r_score), float(center_t), "r_bar",
                        )
                    _, r_roi, r_present = self.r_bar.score(center, det.bar)
                    level_payload["r_bar_roi"] = list(r_roi)
                    level_payload["r_bar_present"] = r_present
                if self.level_reader is not None and det is not None:
                    center = clip[len(clip) // 2]
                    dbg = self.level_reader.read_debug(center, det.bar)
                    if dbg.get("box") is not None:
                        x, y, w, h = dbg["box"]
                        level_payload["level_roi"] = [x, y, w, h]
                    hit = self.level_reader.read(center, det.bar)
                    if hit is not None:
                        level, conf = hit
                        stable = self._level_stabilizer.push(
                            level, conf, min_conf=self.level_reader.match_threshold)
                        if stable is not None:
                            level_payload["level"] = stable
                            level_payload["level_confidence"] = round(conf, 2)
                    elif dbg.get("scores"):
                        best = max(dbg["scores"], key=dbg["scores"].get)
                        level_payload["level_read_miss"] = (
                            f"best {best}@{dbg['scores'][best]:.2f}"
                        )

                if self.on_status and (now - last_status) >= self.status_interval_sec:
                    last_status = now
                    best = max(self.detector.ability_idx, key=lambda i: probs[i])
                    payload = {
                        "champion": self.champion,
                        "top1": self.classes[best],
                        "top1_score": float(probs[best]),
                        "capture_fps": float(grabber.capture_fps),
                        "loc_ms": float(loc_ms),
                        "model_ms": float(model_ms),
                    }
                    payload.update(level_payload)
                    self.on_status(payload)
        finally:
            grabber.stop()
            grabber.join(timeout=1.0)
