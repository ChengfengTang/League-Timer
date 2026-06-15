"""Read/write the timestamp annotation files produced by the annotator.

Schema (one JSON file per source video)::

    {
      "video": "game1.mp4",              # filename, relative to data/{ChampionName}/raw_videos
      "fps": 60.0,
      "events": [
        {"time": 12.34, "frame": 740, "ability": "Q"},
        ...
      ]
    }
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass
class Event:
    time: float       # seconds into the video at the cast moment
    frame: int        # source frame index (native fps)
    ability: str      # one of the config's ability classes

    def to_dict(self) -> dict:
        return {"time": round(self.time, 4), "frame": int(self.frame), "ability": self.ability}


@dataclass
class Annotation:
    video: str
    fps: float
    events: List[Event]

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "fps": self.fps,
            "events": [e.to_dict() for e in sorted(self.events, key=lambda e: e.time)],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Annotation":
        with open(path, "r") as f:
            data = json.load(f)
        events = [
            Event(time=float(e["time"]), frame=int(e.get("frame", 0)), ability=str(e["ability"]))
            for e in data.get("events", [])
        ]
        return cls(video=data["video"], fps=float(data.get("fps", 0.0)), events=events)

    @classmethod
    def load_or_new(cls, path: str | Path, video: str, fps: float) -> "Annotation":
        path = Path(path)
        if path.exists():
            return cls.load(path)
        return cls(video=video, fps=fps, events=[])


def annotation_path_for(video_path: str | Path, annotations_dir: str | Path) -> Path:
    """Default annotation file path for a given video."""
    video_path = Path(video_path)
    return Path(annotations_dir) / f"{video_path.stem}.json"
