"""Find tracked champions on screen via name templates + healthbars.

Identity is **champion name + healthbar**, not bar colour. Allies (green bar) and
enemies (red bar) — or teammates you want to track — use the same name template
flow.

Pipeline per frame (input is RGB, matching ``src.common.video``):

When ``name_templates`` are configured (default):

1. Find champion-shaped healthbars (red **or** green).
2. Template-match the champion name in the band **above** each bar.
3. Keep only bar+name pairs that match a tracked champion; crop under the bar.

A full-frame name search is not used — it produced texture false-positives.

Fallback (no name match): threshold red and/or green champion healthbars,
filter by shape, and confirm each bar by name.

Returns a list of :class:`Detection` for multi-champion tracking.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class Detection:
    champion: str          # e.g. "Ezreal"
    box: Box               # champion crop region (clamped to frame)
    bar: Box               # estimated healthbar bbox below the nameplate
    score: float           # name-template match confidence


# HSV ranges (OpenCV H in [0,180]). Red wraps the hue circle.
_DEFAULT_RED = [[0, 120, 90], [10, 255, 255], [170, 120, 90], [180, 255, 255]]
_DEFAULT_GREEN = [[35, 100, 90], [85, 255, 255]]


class Localizer:
    def __init__(
        self,
        *,
        box_size: int = 220,
        offset_y: int = 5,
        bar_min_w: int = 55,
        bar_max_w: int = 220,
        bar_min_h: int = 4,
        bar_max_h: int = 22,
        bar_min_aspect: float = 5.0,
        red_hsv: Optional[List[List[int]]] = None,
        green_hsv: Optional[List[List[int]]] = None,
        detect_green: bool = False,
        ignore_regions: Optional[List[List[float]]] = None,
        assume_single_enemy: bool = False,
        name_first: bool = True,
        name_templates: Optional[Dict[str, np.ndarray]] = None,
        name_match_threshold: float = 0.6,
        name_scales: Optional[List[float]] = None,
        default_champion: str = "Ezreal",
    ):
        self.box_size = box_size
        self.offset_y = offset_y
        self.bar_min_w = bar_min_w
        self.bar_max_w = bar_max_w
        self.bar_min_h = bar_min_h
        self.bar_max_h = bar_max_h
        self.bar_min_aspect = bar_min_aspect
        self.red_hsv = red_hsv or _DEFAULT_RED
        self.green_hsv = green_hsv or _DEFAULT_GREEN
        self.detect_green = detect_green
        self.ignore_regions = ignore_regions or []
        self.assume_single_enemy = assume_single_enemy
        self.name_first = name_first
        self.name_templates = name_templates or {}
        self.name_match_threshold = name_match_threshold
        self.name_scales = name_scales or [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
        self.default_champion = default_champion

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_config(cls, cfg_localize: Dict, base_dir: Path | str = ".") -> "Localizer":
        base_dir = Path(base_dir)
        templates: Dict[str, np.ndarray] = {}
        for champ, rel in (cfg_localize.get("name_templates") or {}).items():
            p = base_dir / rel
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"name template not found for {champ}: {p}")
            templates[champ] = img
        return cls(
            box_size=int(cfg_localize.get("box_size", 220)),
            offset_y=int(cfg_localize.get("offset_y", 5)),
            bar_min_w=int(cfg_localize.get("bar_min_w", 55)),
            bar_max_w=int(cfg_localize.get("bar_max_w", 220)),
            bar_min_h=int(cfg_localize.get("bar_min_h", 4)),
            bar_max_h=int(cfg_localize.get("bar_max_h", 22)),
            bar_min_aspect=float(cfg_localize.get("bar_min_aspect", 5.0)),
            red_hsv=cfg_localize.get("red_hsv"),
            green_hsv=cfg_localize.get("green_hsv"),
            detect_green=bool(cfg_localize.get("detect_green", False)),
            ignore_regions=cfg_localize.get("ignore_regions"),
            assume_single_enemy=bool(cfg_localize.get("assume_single_enemy", False)),
            name_first=bool(cfg_localize.get("name_first", True)),
            name_templates=templates,
            name_match_threshold=float(cfg_localize.get("name_match_threshold", 0.6)),
            name_scales=cfg_localize.get("name_scales"),
            default_champion=str(cfg_localize.get("default_champion", "Ezreal")),
        )

    # -- detection ---------------------------------------------------------- #
    def _hsv_mask(self, rgb: np.ndarray, ranges: List[List[int]]) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lo, hi = ranges
        return cv2.inRange(hsv, tuple(lo), tuple(hi))

    def _bar_mask(self, rgb: np.ndarray) -> np.ndarray:
        lo1, hi1, lo2, hi2 = self.red_hsv
        mask = cv2.bitwise_or(
            cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), tuple(lo1), tuple(hi1)),
            cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), tuple(lo2), tuple(hi2)),
        )
        if self.detect_green:
            mask = cv2.bitwise_or(mask, self._hsv_mask(rgb, self.green_hsv))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    def _in_ignore_region(self, box: Box, frame_w: int, frame_h: int) -> bool:
        x, y, w, h = box
        cx, cy = (x + w / 2) / frame_w, (y + h / 2) / frame_h
        for rx, ry, rw, rh in self.ignore_regions:
            if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
                return True
        return False

    def detect_bars(self, rgb: np.ndarray) -> List[Box]:
        H, W = rgb.shape[:2]
        mask = self._bar_mask(rgb)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bars: List[Box] = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if (self.bar_min_w <= w <= self.bar_max_w
                    and self.bar_min_h <= h <= self.bar_max_h
                    and w / float(h) >= self.bar_min_aspect
                    and not self._in_ignore_region((x, y, w, h), W, H)):
                bars.append((x, y, w, h))
        return bars

    def _champion_box(self, bar: Box, frame_w: int, frame_h: int) -> Box:
        x, y, w, h = bar
        cx = x + w // 2
        size = self.box_size
        left = cx - size // 2
        top = y + h + self.offset_y
        left = max(0, min(left, frame_w - 1))
        top = max(0, min(top, frame_h - 1))
        right = min(frame_w, left + size)
        bottom = min(frame_h, top + size)
        return (left, top, right - left, bottom - top)

    def _bar_from_name(self, nx: int, ny: int, nw: int, nh: int) -> Box:
        """Fallback bar estimate when HSV search below the name finds nothing."""
        bar_w = max(int(nw * 1.15), 80)
        bar_h = 11
        bar_x = nx + nw // 2 - bar_w // 2
        bar_y = ny + nh + 3
        return (bar_x, bar_y, bar_w, bar_h)

    def _find_bar_below_name(self, rgb: np.ndarray,
                             nx: int, ny: int, nw: int, nh: int) -> Optional[Box]:
        """Find an actual champion healthbar in the strip just below a name match."""
        H, W = rgb.shape[:2]
        pad_x = 30
        x0 = max(0, nx - pad_x)
        x1 = min(W, nx + nw + pad_x)
        y0 = min(H - 1, ny + nh)
        y1 = min(H, y0 + 34)
        if y1 <= y0 + 2:
            return None
        roi = rgb[y0:y1, x0:x1]
        mask = self._bar_mask(roi)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[Box] = None
        best_w = 0
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            # Distant champions have narrower bars; relax width for this local search.
            if (40 <= bw <= self.bar_max_w
                    and self.bar_min_h <= bh <= self.bar_max_h
                    and bw / float(bh) >= 4.0):
                if bw > best_w:
                    best_w = bw
                    best = (x0 + bx, y0 + by, bw, bh)
        return best

    def _name_candidates(self, gray: np.ndarray,
                         tmpl: np.ndarray) -> List[Tuple[float, int, int, int, int]]:
        """Top name-match location per scale, sorted by score (best first)."""
        H, W = gray.shape[:2]
        cands: List[Tuple[float, int, int, int, int]] = []
        for s in self.name_scales:
            th, tw = int(tmpl.shape[0] * s), int(tmpl.shape[1] * s)
            if th < 4 or tw < 4 or th > H or tw > W:
                continue
            t = cv2.resize(tmpl, (tw, th))
            res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score < self.name_match_threshold:
                continue
            _, _, _, loc = cv2.minMaxLoc(res)
            cands.append((score, loc[0], loc[1], tw, th))
        cands.sort(key=lambda c: c[0], reverse=True)
        return cands

    def _match_name(self, gray: np.ndarray, tmpl: np.ndarray) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        """Return best (score, (x, y, w, h)) for a name template on the frame."""
        cands = self._name_candidates(gray, tmpl)
        if not cands:
            return 0.0, None
        score, nx, ny, nw, nh = cands[0]
        return score, (nx, ny, nw, nh)

    def _name_above_bar(self, rgb: np.ndarray, bar: Box) -> Tuple[str, float]:
        """Match champion name templates in the band above a healthbar."""
        if not self.name_templates:
            return ("?", 0.0)
        x, y, w, h = bar
        H, W = rgb.shape[:2]
        pad_x = int(0.45 * w) + 12
        y1 = y
        y0 = max(0, y - 34)
        x0 = max(0, x - pad_x)
        x1 = min(W, x + w + pad_x)
        if y1 - y0 < 6:
            return ("?", 0.0)
        roi = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        best_champ, best_score = "?", 0.0
        for champ, tmpl in self.name_templates.items():
            for score, nx, ny, nw, nh in self._name_candidates(roi, tmpl):
                name_cx = x0 + nx + nw // 2
                bar_cx = x + w // 2
                if abs(name_cx - bar_cx) > max(w, nw):
                    continue
                if score > best_score:
                    best_champ, best_score = champ, score
                break  # best scale for this template in ROI
        if best_score >= self.name_match_threshold:
            return (best_champ, best_score)
        return ("?", best_score)

    def _locate_bars_with_name(self, rgb: np.ndarray) -> List[Detection]:
        """Primary path: healthbar first, confirm champion name above it."""
        H, W = rgb.shape[:2]
        dets: List[Detection] = []
        for bar in self.detect_bars(rgb):
            champ, score = self._name_above_bar(rgb, bar)
            if champ == "?":
                continue
            dets.append(Detection(champ, self._champion_box(bar, W, H), bar, score))
        return dets

    def _locate_by_name(self, rgb: np.ndarray) -> List[Detection]:
        """Legacy full-frame name search with bar confirmation (fallback)."""
        H, W = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        dets: List[Detection] = []
        for champ, tmpl in self.name_templates.items():
            for score, nx, ny, nw, nh in self._name_candidates(gray, tmpl):
                nbox = (nx, ny, nw, nh)
                if self._in_ignore_region(nbox, W, H):
                    continue
                bar = self._find_bar_below_name(rgb, nx, ny, nw, nh)
                if bar is None or self._in_ignore_region(bar, W, H):
                    continue
                dets.append(Detection(champ, self._champion_box(bar, W, H), bar, score))
                break
        return dets

    def _identify_bar(self, rgb: np.ndarray, bar: Box) -> Tuple[str, float]:
        if not self.name_templates:
            return ("?", 0.0)
        x, y, w, h = bar
        H, W = rgb.shape[:2]
        pad_x, pad_top, pad_bot = int(0.5 * w) + 10, 30, 8
        x0, x1 = max(0, x - pad_x), min(W, x + w + pad_x)
        y0, y1 = max(0, y - pad_top), min(H, y + h + pad_bot)
        roi = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        best_champ, best_score = "?", 0.0
        for champ, tmpl in self.name_templates.items():
            score, _ = self._match_name(roi, tmpl)
            if score > best_score:
                best_champ, best_score = champ, score
        if best_score >= self.name_match_threshold:
            return (best_champ, best_score)
        return ("?", best_score)

    def _locate_by_bars(self, rgb: np.ndarray) -> List[Detection]:
        H, W = rgb.shape[:2]
        bars = self.detect_bars(rgb)
        if not bars:
            return []

        if self.assume_single_enemy and not self.name_templates:
            bar = max(bars, key=lambda b: b[2])
            return [Detection(self.default_champion, self._champion_box(bar, W, H), bar, 1.0)]

        dets: List[Detection] = []
        for bar in bars:
            champ, score = self._identify_bar(rgb, bar)
            if champ == "?":
                continue
            dets.append(Detection(champ, self._champion_box(bar, W, H), bar, score))
        return dets

    def locate_clip(self, clip: np.ndarray) -> Optional[Detection]:
        """Pick one champion box for a (T, H, W, 3) clip (centre frame + neighbours)."""
        t = len(clip)
        center = t // 2
        order = [center]
        for d in range(1, center + 1):
            if center - d >= 0:
                order.append(center - d)
            if center + d < t:
                order.append(center + d)
        for idx in order:
            dets = self.locate(clip[idx])
            if dets:
                return max(dets, key=lambda d: (d.score, d.box[2] * d.box[3]))
        return None

    def locate(self, rgb: np.ndarray) -> List[Detection]:
        if self.name_templates:
            dets = self._locate_bars_with_name(rgb)
            if dets:
                return dets
            if self.name_first:
                dets = self._locate_by_name(rgb)
                if dets:
                    return dets
        return self._locate_by_bars(rgb)
