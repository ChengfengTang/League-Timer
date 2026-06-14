"""Real-time ability recognizer over a live screen capture.

The offline counterpart (``src.infer.recognize``) slides a window over a
recorded file. This module does the same thing against your screen *as you
play*: a background thread grabs frames at the model's ``sample_fps`` into a
rolling buffer spanning one clip, the main loop classifies the most recent
clip every ``stride_sec``, and a streaming detector prints ``CAST W/E/R/...``
the moment an ability fires.

Run::

    python -m src.infer.live --config configs/ezreal.yaml \
        --checkpoint models/ezreal/best.pt

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
from src.infer.recognize import load_model


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

    Frames are grabbed as fast as the target cadence allows, converted to RGB,
    and short-side-resized to ``crop_size`` immediately (cheap, and keeps the
    buffer small). The buffer holds the last ``num_frames`` frames spanning one
    clip; the inference loop snapshots it under a lock.
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
                # Same HUD mask + spatial fit as training, applied per frame so the
                # rolling buffer stays small.
                frame = vu.preprocess_frame(full, self.crop_size,
                                            hud_mask=self.hud_mask, frame_mode=self.frame_mode)
                ts = time.monotonic()
                with self._lock:
                    self._buf.append((ts, frame))
                if self.preview:
                    self._build_preview(full, frame)

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
    cfg = Config.load(config_path)
    icfg = cfg.section("infer")
    device = pick_device(device_str)

    model, ckpt = load_model(checkpoint, device)
    classes = ckpt["classes"]
    num_frames = int(ckpt["num_frames"])
    sample_fps = float(ckpt["sample_fps"])
    crop_size = int(ckpt["crop_size"])
    frame_mode = str(ckpt.get("frame_mode", "center_crop"))
    hud_mask = ckpt.get("hud_mask") or []
    # In preview mode you're tuning the config's mask/letterbox (likely before
    # retraining), so reflect the CONFIG rather than the checkpoint.
    if preview:
        frame_mode = cfg.frame_mode
        hud_mask = cfg.hud_mask
    clip_dur = num_frames / sample_fps
    transform = ClipTransform(crop_size, ckpt["mean"], ckpt["std"], train=False,
                              frame_mode=frame_mode)

    stride = float(stride_sec if stride_sec is not None else icfg.get("stride_sec", 0.25))
    detector = StreamDetector(
        classes,
        thresholds=icfg.get("thresholds", {}) or {},
        default_threshold=float(icfg.get("default_threshold", 0.6)),
        nms_window_sec=float(icfg.get("nms_window_sec", 1.0)),
        peak_only=bool(icfg.get("peak_only", False)),
        min_margin=float(icfg.get("min_margin", 0.0)),
        track=icfg.get("track") or None,
    )

    rect = resolve_region(monitor, region)
    print(f"Device: {device}")
    print(f"Loaded {ckpt['backbone']} ({ckpt.get('champion', '?')}), classes={classes}")
    print(f"Capture region: {rect['width']}x{rect['height']} @ "
          f"({rect['left']},{rect['top']})  | clip={clip_dur:.2f}s "
          f"({num_frames}f @ {sample_fps:g}fps), stride={stride:g}s")
    src = "config (preview tuning)" if preview else "checkpoint"
    print(f"Preprocess: frame_mode={frame_mode}, hud_mask={len(hud_mask)} rect(s) [from {src}]")
    if detector.track:
        print(f"Tracking: {', '.join(detector.track)}")

    for n in range(countdown, 0, -1):
        print(f"  starting in {n}... (switch to your game)", end="\r", flush=True)
        time.sleep(1)
    print(" " * 50, end="\r")
    print("LIVE. Cast abilities in-game; detections print below. Ctrl+C to stop.\n")

    grabber = ScreenGrabber(rect, num_frames, sample_fps, crop_size,
                            hud_mask=hud_mask, frame_mode=frame_mode, preview=preview)
    grabber.start()

    preview_window = "League-Timer: full+mask (left) | model input (right) — press q to quit"
    if preview:
        print("Preview window open: red boxes = HUD mask, right panel = what the "
              "model sees. Tune clip.hud_mask in the config if the boxes don't cover "
              "your HUD. Press q (in the window) or Ctrl+C to stop.")

    next_infer = time.monotonic()
    last_status = 0.0
    try:
        while True:
            now = time.monotonic()
            if now < next_infer:
                if preview and not _show_preview(grabber, preview_window):
                    break
                time.sleep(min(0.005, next_infer - now))
                continue
            next_infer += stride

            if preview and not _show_preview(grabber, preview_window):
                break

            frames, center_t = grabber.snapshot()
            if frames is None:
                next_infer = time.monotonic() + stride  # still buffering
                continue

            clip = np.stack(frames, axis=0)  # (T, H, W, 3) uint8
            x = transform(clip).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            for e in detector.push(center_t, probs):
                print(f"[{e['time']:7.1f}s]  CAST  {e['ability']:<5} ({e['score']:.2f})",
                      flush=True)

            if verbose and (now - last_status) >= 0.5:
                last_status = now
                best = max(detector.ability_idx, key=lambda i: probs[i])
                print(f"   ~ {classes[best]:<5} {probs[best]:.2f} | "
                      f"cap {grabber.capture_fps:4.1f}fps", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        grabber.stop()
        grabber.join(timeout=1.0)
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
