"""Non-model cooldown triggers (configured per champion in YAML).

Generic detectors live here; champion YAML selects which ones are enabled and
supplies templates/thresholds. Add ``src/champions/{name}/`` only when logic
cannot be expressed in config (custom state machines, multi-signal fusion, etc.).
"""
from __future__ import annotations

from typing import Optional

from src.timers.r_bar import RBarDetector, merge_events, scan_video_events

__all__ = ["RBarDetector", "merge_events", "scan_video_events"]


def load_r_bar(cfg, base_dir: str = ".") -> Optional[RBarDetector]:
    """Load ``timers.r_bar`` from a champion config's ``timers`` section."""
    timers = cfg.section("timers") if hasattr(cfg, "section") else (cfg.get("timers") or {})
    raw = timers.get("r_bar") if isinstance(timers, dict) else None
    return RBarDetector.from_config(raw, base_dir=base_dir)
