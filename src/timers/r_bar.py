"""Rising-edge detector for charge icons under a champion health bar.

Used when an ultimate shows stack icons below the mana bar instead of (or in
addition to) classifiable VFX — e.g. Ahri Spirit Rush. Fires once on absent ->
present; charge count changes while icons stay visible do not retrigger.

Configure via ``timers.r_bar`` in ``configs/{champion}.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


@dataclass
class RBarConfig:
    enabled: bool = False
    below_bar: int = 0
    height: int = 24
    width_pad: int = 4
    icon_strip_fraction: float = 0.42
    template_path: Optional[str] = None
    match_threshold: float = 0.50
    name_scales: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35)
    hsv_fallback: bool = True
    hsv: List[List[int]] = None  # type: ignore[assignment]
    min_cyan_ratio: float = 0.015
    refractory_sec: float = 3.0
    rearm_after_absent_sec: float = 1.5

    def __post_init__(self) -> None:
        if self.hsv is None:
            self.hsv = [[85, 60, 100], [105, 255, 255]]


class RBarDetector:
    """Rising-edge detector for R charge icons under a champion health bar."""

    def __init__(self, cfg: RBarConfig, template_gray: Optional[np.ndarray]) -> None:
        self.cfg = cfg
        self._template = template_gray
        self._armed = True
        self._last_emit = -1e9
        self._last_present = -1e9

    @classmethod
    def from_config(cls, raw: Optional[Dict], base_dir: Path | str = ".") -> Optional["RBarDetector"]:
        if not raw or not bool(raw.get("enabled")):
            return None
        cfg = RBarConfig(
            enabled=True,
            below_bar=int(raw.get("below_bar", 0)),
            height=int(raw.get("height", 24)),
            width_pad=int(raw.get("width_pad", 4)),
            icon_strip_fraction=float(raw.get("icon_strip_fraction", 0.42)),
            template_path=str(raw["icon_template"]) if raw.get("icon_template") else None,
            match_threshold=float(raw.get("match_threshold", 0.50)),
            hsv_fallback=bool(raw.get("hsv_fallback", True)),
            hsv=[list(v) for v in (raw.get("hsv") or [[85, 60, 100], [105, 255, 255]])],
            min_cyan_ratio=float(raw.get("min_cyan_ratio", 0.015)),
            refractory_sec=float(raw.get("refractory_sec", 3.0)),
            rearm_after_absent_sec=float(raw.get("rearm_after_absent_sec", 1.5)),
        )
        scales = raw.get("name_scales")
        if scales:
            cfg.name_scales = tuple(float(s) for s in scales)

        tmpl = None
        if cfg.template_path:
            p = Path(base_dir) / cfg.template_path
            tmpl = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if tmpl is None:
                raise FileNotFoundError(f"R bar icon template not found: {p}")
        return cls(cfg, tmpl)

    def roi_box(self, bar: Box, frame_w: int, frame_h: int) -> Box:
        x, y, w, h = bar
        pad = self.cfg.width_pad
        left = max(0, x - pad)
        top = min(frame_h - 1, y + h + self.cfg.below_bar)
        width = min(frame_w - left, w + 2 * pad)
        height = min(frame_h - top, self.cfg.height)
        return left, top, width, height

    def _icon_strip(self, roi_bgr: np.ndarray) -> np.ndarray:
        frac = max(0.15, min(0.7, self.cfg.icon_strip_fraction))
        split = max(1, int(round(roi_bgr.shape[0] * (1.0 - frac))))
        return roi_bgr[split:, :]

    def _template_score(self, strip_gray: np.ndarray) -> float:
        if self._template is None or strip_gray.size == 0:
            return 0.0
        sh, sw = strip_gray.shape[:2]
        best = 0.0
        for s in self.cfg.name_scales:
            th, tw = int(self._template.shape[0] * s), int(self._template.shape[1] * s)
            if th < 4 or tw < 4 or th > sh or tw > sw:
                continue
            t = cv2.resize(self._template, (tw, th))
            res = cv2.matchTemplate(strip_gray, t, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(res.max()))
        return best

    def _cyan_ratio(self, strip_bgr: np.ndarray) -> float:
        if strip_bgr.size == 0:
            return 0.0
        hsv = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2HSV)
        lo = np.array(self.cfg.hsv[0], dtype=np.uint8)
        hi = np.array(self.cfg.hsv[1], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        return float(mask.sum() / 255. / mask.size)

    def score(self, rgb: np.ndarray, bar: Box) -> Tuple[float, Box, bool]:
        """Return (best score, roi box, icons_present)."""
        H, W = rgb.shape[:2]
        x, y, w, h = self.roi_box(bar, W, H)
        if w < 8 or h < 6:
            return 0.0, (x, y, w, h), False
        roi = cv2.cvtColor(rgb[y:y + h, x:x + w], cv2.COLOR_RGB2BGR)
        strip = self._icon_strip(roi)
        strip_gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        t_score = self._template_score(strip_gray)
        cyan_ratio = self._cyan_ratio(strip)
        present = t_score >= self.cfg.match_threshold
        if not present and self.cfg.hsv_fallback:
            present = cyan_ratio >= self.cfg.min_cyan_ratio
        best = t_score if self._template is not None else min(1.0, cyan_ratio / max(self.cfg.min_cyan_ratio, 1e-6))
        return best, (x, y, w, h), present

    def update(self, rgb: np.ndarray, bar: Box, now: float) -> Optional[float]:
        """Return a detection score on R cast (icons absent, then appear), else None."""
        score, _, present = self.score(rgb, bar)
        fired: Optional[float] = None
        if present:
            self._last_present = now
            if self._armed and (now - self._last_emit) >= self.cfg.refractory_sec:
                fired = score
                self._last_emit = now
                self._armed = False
        elif self._last_present > 0 and (now - self._last_present) >= self.cfg.rearm_after_absent_sec:
            # Icons fully gone for a while — ready for the next R cast appearance.
            self._armed = True
        return fired

    def reset(self) -> None:
        """Clear state between videos or detector restarts."""
        self._armed = True
        self._last_emit = -1e9
        self._last_present = -1e9


def scan_video_events(
    video_path: str,
    localizer,
    r_bar: RBarDetector,
    ability: str = "R",
    stride_frames: int = 1,
) -> List[Dict]:
    """Scan a recording frame-by-frame for R cast rising edges (offline inference)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    stride_frames = max(1, int(stride_frames))
    r_bar.reset()
    events: List[Dict] = []
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f % stride_frames == 0:
            t = f / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = localizer.locate(rgb)
            if dets:
                det = max(dets, key=lambda d: (d.score, d.box[2] * d.box[3]))
                score = r_bar.update(rgb, det.bar, t)
                if score is not None:
                    events.append({
                        "ability": ability,
                        "time": round(t, 3),
                        "score": round(float(score), 4),
                        "source": "r_bar",
                    })
        f += 1
    cap.release()
    return events


def merge_events(model_events: List[Dict], r_bar_events: List[Dict]) -> List[Dict]:
    """Combine model + r_bar events, sorted by time."""
    out = list(model_events) + list(r_bar_events)
    out.sort(key=lambda e: e["time"])
    return out
