"""Video reading helpers built on OpenCV.

Centralises the messy parts of frame extraction (fps handling, resampling a
window to a fixed number of frames) so the dataset builder and the recognizer
sample clips identically.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def probe(path: str | Path) -> VideoMeta:
    """Read basic metadata for a video file."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if fps <= 0:
        raise ValueError(f"Video reports invalid fps ({fps}): {path}")
    return VideoMeta(path=path, fps=fps, frame_count=frame_count, width=width, height=height)


@lru_cache(maxsize=16)
def last_readable_frame(path: str) -> int:
    """Last frame index OpenCV can actually decode (often < reported frame count)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported <= 0:
            return 0
        lo, hi = 0, reported
        while lo < hi:
            mid = (lo + hi) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ok, frame = cap.read()
            if ok and frame is not None:
                lo = mid + 1
            else:
                hi = mid
        return max(0, lo - 1)
    finally:
        cap.release()


def effective_duration_sec(path: str | Path) -> float:
    """Readable duration in seconds (may be shorter than probe().duration_sec)."""
    meta = probe(path)
    return last_readable_frame(str(Path(path))) / meta.fps


def resize_short_side(frame: np.ndarray, target: int) -> np.ndarray:
    """Resize so the shorter spatial side equals ``target``, preserving aspect."""
    h, w = frame.shape[:2]
    if min(h, w) == target:
        return frame
    if h <= w:
        new_h = target
        new_w = max(1, int(round(w * target / h)))
    else:
        new_w = target
        new_h = max(1, int(round(h * target / w)))
    interp = cv2.INTER_AREA if target < min(h, w) else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h), interpolation=interp)


def _read_frame_at(cap: "cv2.VideoCapture", frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    # OpenCV returns BGR; convert to RGB for downstream torchvision transforms.
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def sample_clip(
    path: str | Path,
    center_sec: float,
    num_frames: int,
    sample_fps: float,
    resize_short: int | None = None,
) -> np.ndarray:
    """Sample ``num_frames`` RGB frames spanning a window centred on ``center_sec``.

    The window has duration ``num_frames / sample_fps`` seconds. Frames are
    resampled from the source video's native fps to the requested ``sample_fps``.
    Out-of-range frames are clamped to the nearest valid frame so callers always
    get a full clip. If ``resize_short`` is set, each frame's shorter side is
    resized to that many pixels (aspect preserved).

    Returns array of shape (num_frames, H, W, 3), dtype uint8.
    """
    meta = probe(path)
    max_frame = last_readable_frame(str(path))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        span_sec = num_frames / sample_fps
        start_sec = center_sec - span_sec / 2.0
        frames: List[np.ndarray] = []
        last_good: np.ndarray | None = None
        for i in range(num_frames):
            t = start_sec + (i + 0.5) / sample_fps
            frame_idx = int(round(t * meta.fps))
            frame_idx = min(max(frame_idx, 0), max_frame)
            frame = _read_frame_at(cap, frame_idx)
            if frame is None:
                frame = last_good
            if frame is not None:
                last_good = frame
                if resize_short is not None:
                    frame = resize_short_side(frame, resize_short)
            frames.append(frame)
        # Backfill any leading None frames with the first good frame.
        first_good = next((f for f in frames if f is not None), None)
        if first_good is None:
            raise RuntimeError(f"Could not read any frames near {center_sec:.2f}s in {path}")
        frames = [f if f is not None else first_good for f in frames]
        return np.stack(frames, axis=0)
    finally:
        cap.release()
