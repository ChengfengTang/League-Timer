"""Config loading and small typed accessors shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class Config:
    """Thin wrapper over the per-champion YAML config.

    Keeps the raw dict accessible while exposing the few values that several
    stages need so call sites don't repeat dictionary digging.
    """

    path: Path
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls(path=path, raw=raw)
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        if not self.classes:
            raise ValueError(f"{self.path}: 'classes' must be a non-empty list")
        if "background" not in self.classes:
            raise ValueError(f"{self.path}: 'classes' must include 'background'")
        if self.classes[0] != "background":
            # Keeping background at index 0 makes argmax==0 mean "nothing".
            raise ValueError(f"{self.path}: 'background' must be the first class")

    # --- convenience accessors ---------------------------------------------
    @property
    def champion(self) -> str:
        return str(self.raw.get("champion", self.path.stem))

    @property
    def classes(self) -> List[str]:
        return list(self.raw.get("classes", []))

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def class_to_id(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.classes)}

    def id_to_class(self) -> Dict[int, str]:
        return {i: name for i, name in enumerate(self.classes)}

    @property
    def ability_classes(self) -> List[str]:
        """All classes except background (i.e. the things we want to detect)."""
        return [c for c in self.classes if c != "background"]

    @property
    def num_frames(self) -> int:
        return int(self.raw.get("clip", {}).get("num_frames", 13))

    @property
    def sample_fps(self) -> float:
        return float(self.raw.get("clip", {}).get("sample_fps", 13))

    @property
    def crop_size(self) -> int:
        return int(self.raw.get("clip", {}).get("crop_size", 182))

    @property
    def frame_mode(self) -> str:
        """How a frame is fit to the square model input: 'letterbox' | 'center_crop'."""
        return str(self.raw.get("clip", {}).get("frame_mode", "letterbox"))

    @property
    def hud_mask(self) -> List[List[float]]:
        """Rects [x, y, w, h] (fractions) blacked out before use; [] if none."""
        return [list(r) for r in (self.raw.get("clip", {}).get("hud_mask", []) or [])]

    @property
    def spatial_jitter(self) -> float:
        """Train-time random zoom/reposition strength (0 disables)."""
        return float(self.raw.get("clip", {}).get("spatial_jitter", 0.0))

    @property
    def clip_duration_sec(self) -> float:
        """Wall-clock duration spanned by one clip."""
        return self.num_frames / self.sample_fps

    @property
    def localize_enabled(self) -> bool:
        return bool(self.raw.get("localize", {}).get("enabled", False))

    def section(self, name: str) -> Dict[str, Any]:
        return dict(self.raw.get(name, {}))

    # --- per-champion data layout (data/{ChampionName}/...) ----------------
    def data_dir(self, base: str = "data") -> Path:
        return Path(base) / self.champion

    def raw_videos_dir(self, base: str = "data") -> Path:
        return self.data_dir(base) / "raw_videos"

    def annotations_dir(self, base: str = "data") -> Path:
        return self.data_dir(base) / "annotations"

    def clips_dir(self, base: str = "data") -> Path:
        return self.data_dir(base) / "clips"
