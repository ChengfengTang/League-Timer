"""Read champion level from the badge left of the healthbar.

Uses template matching on a small ROI relative to the localized healthbar.
Digit templates load from ``configs/templates/level_digits/{0-9}.png`` when
present; otherwise falls back to built-in bitmap templates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]

# Compact 7x9 bitmaps for digits 0-9 (fallback when PNG templates are absent).
_BITMAP_7X9: Dict[str, List[str]] = {
    "0": [
        ".#####.",
        "#.....#",
        "#.....#",
        "#.....#",
        "#.....#",
        "#.....#",
        "#.....#",
        "#.....#",
        ".#####.",
    ],
    "1": [
        "..#..",
        ".##..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        ".###.",
    ],
    "2": [
        ".#####.",
        "#.....#",
        "......#",
        ".....#.",
        "...##..",
        "..#....",
        ".#.....",
        "#......",
        "#######",
    ],
    "3": [
        "#####.",
        ".....#",
        ".....#",
        ".....#",
        "#####.",
        ".....#",
        ".....#",
        ".....#",
        "#####.",
    ],
    "4": [
        "#....#",
        "#....#",
        "#....#",
        "#....#",
        "######",
        ".....#",
        ".....#",
        ".....#",
        ".....#",
    ],
    "5": [
        "#######",
        "#......",
        "#......",
        "#......",
        "######.",
        "......#",
        "......#",
        "......#",
        "######.",
    ],
    "6": [
        ".#####.",
        "#......",
        "#......",
        "#......",
        "######.",
        "#.....#",
        "#.....#",
        "#.....#",
        ".#####.",
    ],
    "7": [
        "#######",
        ".....#.",
        "....#..",
        "...#...",
        "..#....",
        ".#.....",
        ".#.....",
        ".#.....",
        ".#.....",
    ],
    "8": [
        ".#####.",
        "#.....#",
        "#.....#",
        "#.....#",
        ".#####.",
        "#.....#",
        "#.....#",
        "#.....#",
        ".#####.",
    ],
    "9": [
        ".#####.",
        "#.....#",
        "#.....#",
        "#.....#",
        ".######",
        "......#",
        "......#",
        "......#",
        ".#####.",
    ],
}


def _bitmap_to_gray(pattern: List[str], scale: int = 3) -> np.ndarray:
    rows = []
    for line in pattern:
        row = [255 if ch == "#" else 0 for ch in line]
        rows.append(row)
    arr = np.array(rows, dtype=np.uint8)
    if scale > 1:
        arr = cv2.resize(arr, (arr.shape[1] * scale, arr.shape[0] * scale),
                         interpolation=cv2.INTER_NEAREST)
    return arr


def _load_digit_templates(base_dir: Path | str) -> Dict[str, np.ndarray]:
    base = Path(base_dir)
    out = {d: _bitmap_to_gray(_BITMAP_7X9[d]) for d in "0123456789"}
    for d in "0123456789":
        path = base / f"{d}.png"
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            out[d] = img
    return out


class LevelReader:
    def __init__(
        self,
        *,
        offset_x: int = -28,
        offset_y: int = -1,
        width: int = 28,
        height: int = 20,
        match_threshold: float = 0.55,
        digit_templates_dir: str = "configs/templates/level_digits",
        scales: Optional[List[float]] = None,
    ) -> None:
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.width = width
        self.height = height
        self.match_threshold = match_threshold
        self.scales = scales or [0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25]
        self._digits = _load_digit_templates(digit_templates_dir)

    @classmethod
    def from_config(cls, cfg: Dict, base_dir: Path | str = ".") -> "LevelReader":
        base = Path(base_dir)
        tpl = str(cfg.get("digit_templates", "configs/templates/level_digits"))
        if not Path(tpl).is_absolute():
            tpl = str(base / tpl)
        return cls(
            offset_x=int(cfg.get("offset_x", -28)),
            offset_y=int(cfg.get("offset_y", -1)),
            width=int(cfg.get("width", 28)),
            height=int(cfg.get("height", 20)),
            match_threshold=float(cfg.get("match_threshold", 0.55)),
            digit_templates_dir=tpl,
            scales=cfg.get("scales"),
        )

    def level_box(self, rgb: np.ndarray, bar: Box) -> Optional[Box]:
        """Return level-badge ROI as ``(x, y, w, h)`` in frame pixels."""
        H, W = rgb.shape[:2]
        bx, by, bw, bh = bar
        x0 = max(0, bx + self.offset_x)
        y0 = max(0, by + self.offset_y)
        x1 = min(W, x0 + self.width)
        y1 = min(H, y0 + self.height)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def _level_roi(self, rgb: np.ndarray, bar: Box) -> Optional[np.ndarray]:
        box = self.level_box(rgb, bar)
        if box is None:
            return None
        x0, y0, w, h = box
        return rgb[y0:y0 + h, x0:x0 + w]

    def _isolate_digit(self, gray: np.ndarray) -> np.ndarray:
        """Crop to bright digit pixels inside the level badge."""
        _, bright = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)
        pts = cv2.findNonZero(bright)
        if pts is None:
            return gray
        x, y, w, h = cv2.boundingRect(pts)
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(gray.shape[1], x + w + pad)
        y1 = min(gray.shape[0], y + h + pad)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return gray
        return gray[y0:y1, x0:x1]

    def _prep_gray(self, roi_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        gray = self._isolate_digit(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = cv2.countNonZero(binary) / max(binary.size, 1)
        if bright > 0.5:
            binary = cv2.bitwise_not(binary)
        return binary

    def _prep_gray_raw(self, roi_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        return self._isolate_digit(gray)

    def _normalize_for_match(self, gray: np.ndarray, height: int = 24) -> np.ndarray:
        h, w = gray.shape[:2]
        if h < 2 or w < 2:
            return gray
        nh = height
        nw = max(4, int(round(w * nh / h)))
        return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

    def _fit_template(self, gray_n: np.ndarray, tmpl_n: np.ndarray) -> np.ndarray:
        """Shrink *tmpl_n* to fit inside *gray_n* when the normalized template is larger."""
        gh, gw = gray_n.shape[:2]
        th, tw = tmpl_n.shape[:2]
        if th <= gh and tw <= gw:
            return tmpl_n
        scale = min(gw / max(tw, 1), gh / max(th, 1)) * 0.98
        if scale < 0.35:
            return tmpl_n
        return cv2.resize(
            tmpl_n,
            (max(4, int(round(tw * scale))), max(4, int(round(th * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def _match_digit(self, gray: np.ndarray, digit: str) -> float:
        tmpl = self._digits[digit]
        gray_n = self._normalize_for_match(gray)
        tmpl_n = self._fit_template(gray_n, self._normalize_for_match(tmpl))
        gh, gw = gray_n.shape[:2]
        th, tw = tmpl_n.shape[:2]
        if th < 4 or tw < 4 or th > gh or tw > gw:
            return 0.0
        res = cv2.matchTemplate(gray_n, tmpl_n, cv2.TM_CCOEFF_NORMED)
        base = float(res.max())
        best = base
        for s in self.scales:
            if abs(s - 1.0) < 1e-6:
                continue
            th, tw = int(tmpl_n.shape[0] * s), int(tmpl_n.shape[1] * s)
            if th < 4 or tw < 4 or th > gh or tw > gw:
                continue
            t = cv2.resize(tmpl_n, (tw, th), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(gray_n, t, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(res.max()))
        return best

    def _best_digit(self, gray: np.ndarray) -> Tuple[Optional[str], float]:
        best_d, best_s = None, 0.0
        for d in "0123456789":
            score = self._match_digit(gray, d)
            if score > best_s:
                best_d, best_s = d, score
        return best_d, best_s

    def _read_two_digits(self, gray: np.ndarray) -> Optional[Tuple[str, float]]:
        """Try several split points and return the best valid level-10–18 decode."""
        w = gray.shape[1]
        narrow_max = max(4, w // 4)
        candidates: List[Tuple[str, float, int]] = []
        lo = max(2, w // 5)
        hi = min(w - 2, 3 * w // 4 + 1)
        for mid in range(lo, hi):
            left, right = gray[:, :mid], gray[:, mid:]
            left_best = self._read_digits(left, 1)
            if not left_best or left_best[0] != "1":
                continue
            right_best = self._read_digits(right, 1)
            if not right_best:
                continue
            text = left_best[0] + right_best[0]
            if not text.isdigit():
                continue
            level = int(text)
            if not (10 <= level <= 18):
                continue
            score = min(left_best[1], right_best[1])
            if score < self.match_threshold:
                continue
            candidates.append((text, score, mid))
        if not candidates:
            return None
        narrow = [c for c in candidates if c[2] <= narrow_max]
        pool = narrow if narrow else candidates
        text, score, _ = max(pool, key=lambda c: (c[1], -c[2]))
        return text, score

    def _read_digits(self, gray: np.ndarray, n: int) -> Optional[Tuple[str, float]]:
        if n == 1:
            best_d, best_s = self._best_digit(gray)
            if best_d and best_s >= self.match_threshold:
                return best_d, best_s
            return None

        return self._read_two_digits(gray)

    def read_debug(self, rgb: np.ndarray, bar: Box) -> Dict:
        """Return level read attempt details for preview / tuning."""
        box = self.level_box(rgb, bar)
        if box is None:
            return {"box": None, "level": None, "confidence": 0.0, "scores": {}}
        roi = self._level_roi(rgb, bar)
        scores: Dict[str, float] = {}
        level: Optional[int] = None
        confidence = 0.0
        if roi is not None:
            gray = self._prep_gray(roi)
            for d in "0123456789":
                scores[d] = round(self._match_digit(gray, d), 3)
            hit = self.read(rgb, bar)
            if hit is not None:
                level, confidence = hit
        return {
            "box": box,
            "level": level,
            "confidence": round(confidence, 3),
            "scores": scores,
        }

    def read(self, rgb: np.ndarray, bar: Box) -> Optional[Tuple[int, float]]:
        """Return ``(level, confidence)`` or ``None`` if unreadable."""
        roi = self._level_roi(rgb, bar)
        if roi is None:
            return None
        gray = self._prep_gray_raw(roi)
        w = gray.shape[1]
        if w >= 12:
            hit2 = self._read_two_digits(gray)
            if hit2 is not None:
                level2 = int(hit2[0])
                if 10 <= level2 <= 18 and hit2[1] >= self.match_threshold:
                    return level2, hit2[1]
        hit1 = self._read_digits(gray, 1)
        if hit1 is not None:
            level1 = int(hit1[0])
            if 1 <= level1 <= 9 and hit1[1] >= self.match_threshold:
                return level1, hit1[1]
        return None
