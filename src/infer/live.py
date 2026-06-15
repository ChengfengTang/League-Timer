"""Real-time ability recognizer over a live screen capture.

The offline counterpart (``src.infer.recognize``) slides a window over a
recorded file. This module does the same thing against your screen *as you
play*: a background thread grabs frames at the model's ``sample_fps`` into a
rolling buffer spanning one clip, the main loop classifies the most recent
clip every ``stride_sec``, and a streaming detector prints ``CAST W/E/R/...``
the moment an ability fires.

Run::

    python -m src.infer.live --config configs/{ChampionName}.yaml \
        --checkpoint models/{ChampionName}/best.pt

macOS note: the first run will prompt for Screen Recording permission
(System Settings -> Privacy & Security -> Screen Recording). Grant it to your
terminal / IDE and restart the app.

Preprocessing is kept identical to training/offline inference: each captured
frame has its short side resized to ``crop_size`` (RGB), then ``ClipTransform``
center-crops + normalizes the stacked clip. Detection reuses the same
threshold / min-margin / peak / NMS logic from the recognizer, adapted to a
streaming buffer so events can be emitted with bounded latency.
"""
from __future__ import annotations

import os

# The x3d backbone uses avg_pool3d, which MPS doesn't implement; allow the op to
# fall back to CPU. Must be set before torch is first imported (below / transitively).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.common import video as vu
from src.common.config import Config
from src.common.torch_utils import pick_device
from src.dataset.clip_dataset import ClipTransform
from src.infer.recognize import _localizer_from_ckpt, load_model, prepare_model_clip
from src.localize.localizer import Detection


def _live_cfg(icfg: Dict) -> Dict:
    """Merge ``infer.live`` overrides on top of shared ``infer`` settings."""
    base = dict(icfg)
    overrides = icfg.get("live") or {}
    for key, val in overrides.items():
        if key == "thresholds":
            merged = dict(base.get("thresholds") or {})
            merged.update(val or {})
            base["thresholds"] = merged
        else:
            base[key] = val
    return base


def _prepare_live_clip(
    clip: np.ndarray,
    ckpt: Dict,
    localizer,
    cache: Dict,
) -> Tuple[Optional[np.ndarray], float]:
    """Localize + crop for live, with a short-lived box cache for speed.

    Returns (model_clip, localize_ms). On cache refresh we only search the
    centre frame — ``locate_clip`` over 13 full-res frames is what causes
    occasional 1–2s inference spikes (e.g. right after a big R VFX).
    """
    crop_size = int(ckpt["crop_size"])
    if localizer is None:
        t0 = time.monotonic()
        out = prepare_model_clip(clip, ckpt, None)
        return out, (time.monotonic() - t0) * 1000.0

    t0 = time.monotonic()
    now = time.monotonic()
    cache_sec = float(cache.get("sec", 0.0))
    det: Detection | None = None
    cached = cache.get("det")
    cached_t = float(cache.get("t", 0.0))
    if cached is not None and cache_sec > 0 and (now - cached_t) < cache_sec:
        det = cached
    else:
        # Fast refresh: one frame only. Full locate_clip scans up to 13×1080p.
        center = clip[len(clip) // 2]
        dets = localizer.locate(center)
        if dets:
            det = max(dets, key=lambda d: (d.score, d.box[2] * d.box[3]))
            cache["det"] = det
            cache["t"] = now
        elif cached is not None:
            det = cached
    loc_ms = (time.monotonic() - t0) * 1000.0
    if det is None:
        return None, loc_ms
    return vu.crop_clip_to_box(clip, det.box, crop_size), loc_ms


def _open_mss():
    """Open an mss capture handle, preferring the non-deprecated ``MSS`` class."""
    import mss

    factory = getattr(mss, "MSS", None) or mss.mss
    return factory()


# --------------------------------------------------------------------------- #
# Screen capture                                                              #
# --------------------------------------------------------------------------- #
class ScreenGrabber(threading.Thread):
    """Background thread that fills a rolling buffer of recent frames.

    Frames are grabbed at the target cadence and stored as full-resolution RGB
    (localization + cropping happen at inference time, matching offline recognize).
    """

    def __init__(self, region: Dict[str, int], num_frames: int, sample_fps: float,
                 crop_size: int, hud_mask: Optional[List[List[float]]] = None,
                 frame_mode: str = "letterbox", preview: bool = False,
                 preview_h: int = 480):
        super().__init__(daemon=True)
        self.region = region
        self.crop_size = crop_size
        self.hud_mask = hud_mask or []
        self.frame_mode = frame_mode
        self.preview = preview
        self.preview_h = preview_h
        self.target_interval = 1.0 / sample_fps
        self._buf: Deque[Tuple[float, np.ndarray]] = deque(maxlen=num_frames)
        self._lock = threading.Lock()
        # NB: don't name this ``_stop`` — that shadows Thread._stop() and breaks join().
        self._stop_event = threading.Event()
        self._fps_ema: Optional[float] = None
        self.start_mono = time.monotonic()
        # Latest (full_with_mask, model_input) RGB frames for the optional preview.
        self._preview: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def capture_fps(self) -> float:
        return self._fps_ema or 0.0

    def snapshot(self) -> Tuple[Optional[List[np.ndarray]], float]:
        """Return (frames, center_time) if the buffer is full, else (None, t)."""
        with self._lock:
            if len(self._buf) < self._buf.maxlen:
                return None, time.monotonic() - self.start_mono
            frames = [f for _, f in self._buf]
            center = (self._buf[0][0] + self._buf[-1][0]) / 2.0 - self.start_mono
        return frames, center

    def preview_images(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Latest (full-frame-with-mask, model-input) RGB pair, or None yet."""
        with self._lock:
            return self._preview

    def _build_preview(self, full_rgb: np.ndarray, model_input: np.ndarray) -> None:
        """Draw the HUD mask onto a downscaled full frame and stash for display."""
        disp = vu.resize_short_side(full_rgb, self.preview_h).copy()
        h, w = disp.shape[:2]
        for r in self.hud_mask:
            fx, fy, fw, fh = (float(v) for v in r)
            x0, y0 = int(round(fx * w)), int(round(fy * h))
            x1, y1 = int(round((fx + fw) * w)), int(round((fy + fh) * h))
            disp[max(0, y0):y1, max(0, x0):x1] = 0          # masked-out region
            cv2.rectangle(disp, (x0, y0), (x1, y1), (255, 0, 0), 2)  # red outline (RGB)
        with self._lock:
            self._preview = (disp, model_input.copy())

    def run(self) -> None:
        # mss must be created inside the thread that uses it (esp. on macOS).
        last_tick = time.monotonic()
        last_t = last_tick
        with _open_mss() as sct:
            while not self._stop_event.is_set():
                now = time.monotonic()
                sleep_for = self.target_interval - (now - last_tick)
                if sleep_for > 0:
                    time.sleep(sleep_for)
                last_tick = time.monotonic()

                raw = np.asarray(sct.grab(self.region))  # BGRA, (H, W, 4)
                full = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
                ts = time.monotonic()
                with self._lock:
                    self._buf.append((ts, full))
                if self.preview:
                    # Model-input preview is built in the main loop once localized.
                    self._build_preview(full, full)

                dt = ts - last_t
                last_t = ts
                if dt > 0:
                    inst = 1.0 / dt
                    self._fps_ema = inst if self._fps_ema is None else \
                        0.9 * self._fps_ema + 0.1 * inst


# --------------------------------------------------------------------------- #
# Streaming event detection                                                   #
# --------------------------------------------------------------------------- #
class StreamDetector:
    """Streaming version of ``recognize.detect_events``.

    Windows are buffered and only finalized once enough "future" has arrived to
    judge a local peak (``peak_radius`` seconds). For each finalized window we
    apply the same threshold / min-margin / peak filters, then a per-ability
    refractory period (``nms_window_sec``) so a single cast emits once.
    """

    def __init__(self, classes: List[str], thresholds: Dict[str, float],
                 default_threshold: float, nms_window_sec: float,
                 peak_only: bool, min_margin: float, track: Optional[List[str]]):
        self.classes = classes
        self.thresholds = thresholds or {}
        self.default_threshold = default_threshold
        self.nms_window_sec = nms_window_sec
        self.peak_only = peak_only
        self.min_margin = min_margin
        self.track = track
        self.peak_radius = max(nms_window_sec / 2.0, 0.05)
        self.ability_idx = [i for i, c in enumerate(classes) if c != "background"]

        self._hist: List[Tuple[float, np.ndarray]] = []
        self._cursor = 0
        self._last_emit: Dict[str, float] = {}
        self._latest_t = 0.0

    def push(self, t: float, probs: np.ndarray) -> List[Dict]:
        """Add a scored window; return any events finalized by this update."""
        self._hist.append((t, probs))
        self._latest_t = t
        events: List[Dict] = []

        while self._cursor < len(self._hist):
            ct, cp = self._hist[self._cursor]
            if self._latest_t - ct < self.peak_radius:
                break  # not enough future yet to confirm a peak
            events.extend(self._evaluate(self._cursor))
            self._cursor += 1

        self._trim()
        return events

    def _evaluate(self, i: int) -> List[Dict]:
        t, probs = self._hist[i]
        out: List[Dict] = []
        for ci, name in enumerate(self.classes):
            if name == "background":
                continue
            if self.track and name not in self.track:
                continue
            score = float(probs[ci])
            thr = float(self.thresholds.get(name, self.default_threshold))
            if score < thr:
                continue
            if self.min_margin > 0:
                others = [probs[j] for j in self.ability_idx if j != ci]
                if others and (score - max(others)) < self.min_margin:
                    continue
            if self.peak_only and not self._is_peak(i, ci, score, t):
                continue
            last = self._last_emit.get(name)
            if last is not None and (t - last) < self.nms_window_sec:
                continue
            self._last_emit[name] = t
            out.append({"ability": name, "time": round(t, 3),
                        "score": round(score, 4)})
        return out

    def _is_peak(self, i: int, ci: int, score: float, t: float) -> bool:
        for tj, pj in self._hist:
            if abs(tj - t) <= self.peak_radius and float(pj[ci]) > score + 1e-9:
                return False
        return True

    def _trim(self) -> None:
        # Keep enough past for peak comparison + a margin; drop the rest.
        keep_after = self._latest_t - (self.peak_radius + self.nms_window_sec + 1.0)
        drop = 0
        while drop < self._cursor and self._hist[drop][0] < keep_after:
            drop += 1
        if drop:
            self._hist = self._hist[drop:]
            self._cursor -= drop


# --------------------------------------------------------------------------- #
# Region helpers                                                              #
# --------------------------------------------------------------------------- #
def resolve_region(monitor: int, region: Optional[str]) -> Dict[str, int]:
    """Pick the capture rectangle from a monitor index or explicit x,y,w,h."""
    with _open_mss() as sct:
        monitors = sct.monitors  # [0]=all, [1..]=individual
        if region:
            try:
                x, y, w, h = (int(v) for v in region.split(","))
            except ValueError as e:
                raise SystemExit(f"--region must be 'x,y,w,h', got {region!r}") from e
            return {"left": x, "top": y, "width": w, "height": h}
        if monitor < 0 or monitor >= len(monitors):
            raise SystemExit(
                f"--monitor {monitor} out of range (have {len(monitors) - 1} "
                f"monitor(s); 1 = primary, 0 = all combined)")
        m = monitors[monitor]
        return {"left": m["left"], "top": m["top"],
                "width": m["width"], "height": m["height"]}


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #
def _show_preview(grabber: "ScreenGrabber", window: str) -> bool:
    """Render the masked full frame + model input. Returns False if user quit."""
    pv = grabber.preview_images()
    if pv is not None:
        full, model_input = pv
        h = full.shape[0]
        model_up = cv2.resize(model_input, (h, h), interpolation=cv2.INTER_NEAREST)
        gap = np.zeros((h, 8, 3), dtype=full.dtype)
        canvas = np.hstack([full, gap, model_up])
        cv2.imshow(window, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def run(config_path: str, checkpoint: str, device_str: str, monitor: int,
        region: Optional[str], stride_sec: Optional[float], countdown: int,
        verbose: bool, preview: bool = False) -> None:
    # Lazy import avoids a circular import (detector imports building blocks here).
    from src.app.detector import LiveDetector

    det = LiveDetector(
        config_path, checkpoint, device_str=device_str, monitor=monitor,
        region=region, stride_sec=stride_sec,
    )

    # Preview-tuning without a localizer reflects the CONFIG mask/letterbox so you
    # can align clip.hud_mask before retraining (matches the old CLI behaviour).
    if preview and det.localizer is None:
        cfg = Config.load(config_path)
        det.frame_mode = cfg.frame_mode
        det.hud_mask = cfg.hud_mask
        det.transform = ClipTransform(det.crop_size, det.ckpt["mean"], det.ckpt["std"],
                                      train=False, frame_mode=det.frame_mode)

    loc = "localize+crop" if det.localizer else "full-frame"
    print(f"Device: {det.device}")
    print(f"Loaded {det.ckpt['backbone']} ({det.ckpt.get('champion', '?')}), "
          f"classes={det.classes}, preprocess={loc}")
    print(f"Capture region: {det.region['width']}x{det.region['height']} @ "
          f"({det.region['left']},{det.region['top']})  | clip={det.clip_dur:.2f}s "
          f"({det.num_frames}f @ {det.sample_fps:g}fps), stride={det.stride:g}s")
    print(f"Live detect: threshold>={det.detector.default_threshold:g}, "
          f"peak_only={det.detector.peak_only}, min_margin={det.detector.min_margin:g}, "
          f"loc_cache={det.loc_cache['sec']:g}s")
    if det.detector.track:
        print(f"Tracking: {', '.join(det.detector.track)}")

    def on_event(ability: str, score: float, t: float) -> None:
        print(f"[{t:7.1f}s]  CAST  {ability:<5} ({score:.2f})", flush=True)

    def on_status(s: Dict) -> None:
        if not verbose:
            return
        if s.get("localize_miss"):
            print(f"   ! localize miss ({s['loc_ms']:.0f}ms)", end="\r", flush=True)
            return
        print(f"   ~ {s['top1']:<5} {s['top1_score']:.2f} | "
              f"cap {s['capture_fps']:4.1f}fps | "
              f"loc {s['loc_ms']:4.0f}ms model {s['model_ms']:4.0f}ms",
              end="\r", flush=True)

    det.on_event = on_event
    det.on_status = on_status

    for n in range(countdown, 0, -1):
        print(f"  starting in {n}... (switch to your game)", end="\r", flush=True)
        time.sleep(1)
    print(" " * 50, end="\r")
    print("LIVE. Cast abilities in-game; detections print below. Ctrl+C to stop.\n")

    preview_window = "League-Timer: full+mask (left) | model input (right) — press q to quit"
    if preview:
        print("Preview window open: red boxes = HUD mask, right panel = what the "
              "model sees. Tune clip.hud_mask in the config if the boxes don't cover "
              "your HUD. Press q (in the window) or Ctrl+C to stop.")

    preview_cb = (lambda g: _show_preview(g, preview_window)) if preview else None
    try:
        det.run_loop(preview_cb=preview_cb)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if preview:
            cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="Real-time screen-capture ability recognizer.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--monitor", type=int, default=1,
                   help="monitor index (1 = primary, 0 = all combined). Ignored if --region set.")
    p.add_argument("--region", default=None,
                   help="explicit capture rect 'x,y,w,h' (overrides --monitor)")
    p.add_argument("--stride-sec", type=float, default=None,
                   help="seconds between inferences (default: config infer.stride_sec)")
    p.add_argument("--countdown", type=int, default=3,
                   help="seconds to wait before starting (switch to your game)")
    p.add_argument("--verbose", action="store_true",
                   help="show a live top-1 readout + capture fps")
    p.add_argument("--preview", action="store_true",
                   help="open a window showing the masked full frame + model input "
                        "(to tune clip.hud_mask alignment)")
    args = p.parse_args()
    run(args.config, args.checkpoint, args.device, args.monitor, args.region,
        args.stride_sec, args.countdown, args.verbose, args.preview)


if __name__ == "__main__":
    main()
