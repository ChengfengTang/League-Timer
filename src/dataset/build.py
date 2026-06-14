"""Turn timestamp annotations into fixed-length training clips.

For every annotation file it finds, this:
  - cuts one positive clip centred on each labelled cast event, and
  - samples background (negative) clips from windows that don't overlap events.

Clips are saved as ``.npy`` uint8 stacks (T, H, W, 3) so training never has to
re-decode video. A ``manifest.json`` records the class, source, and train/val
split for every clip.

Run::

    python -m src.dataset.build --config configs/ezreal.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from src.common.annotations import Annotation
from src.common.config import Config
from src.common import video as video_utils


def _find_video(video_name: str, raw_dir: Path) -> Optional[Path]:
    candidate = raw_dir / video_name
    if candidate.exists():
        return candidate
    # Fall back to matching by stem (extension may differ).
    stem = Path(video_name).stem
    for p in raw_dir.glob(f"{stem}.*"):
        if p.is_file():
            return p
    return None


def _sample_negative_times(
    duration: float,
    clip_dur: float,
    event_times: List[float],
    n: int,
    min_gap: float,
    rng: random.Random,
) -> List[float]:
    lo = clip_dur / 2.0
    hi = duration - clip_dur / 2.0
    if hi <= lo:
        return []
    out: List[float] = []
    attempts = 0
    max_attempts = n * 50
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        t = rng.uniform(lo, hi)
        if all(abs(t - et) >= min_gap for et in event_times):
            out.append(t)
    return out


def build(config_path: str, raw_dir: str, annotations_dir: str, clips_root: str) -> None:
    cfg = Config.load(config_path)
    ds = cfg.section("dataset")
    rng = random.Random(int(ds.get("seed", 42)))

    raw_dir = Path(raw_dir)
    annotations_dir = Path(annotations_dir)
    out_dir = Path(clips_root) / cfg.champion
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(annotations_dir.glob("*.json"))
    if not ann_files:
        raise SystemExit(f"No annotation files found in {annotations_dir}. Annotate some videos first.")

    valid_abilities = set(cfg.ability_classes)
    clip_dur = cfg.clip_duration_sec
    pos_offset = float(ds.get("positive_offset_sec", 0.0))
    neg_per_video = int(ds.get("negatives_per_video", 40))
    neg_min_gap = float(ds.get("negative_min_gap_sec", 1.0))

    items: List[Dict] = []
    skipped = 0

    for ann_file in ann_files:
        ann = Annotation.load(ann_file)
        video_path = _find_video(ann.video, raw_dir)
        if video_path is None:
            print(f"  ! video not found for {ann_file.name} (looked for {ann.video}); skipping")
            continue
        meta = video_utils.probe(video_path)

        # --- positives ---
        events = [e for e in ann.events if e.ability in valid_abilities]
        skipped += len(ann.events) - len(events)
        centers = [(e.ability, e.time + pos_offset) for e in events]

        # --- negatives ---
        neg_times = _sample_negative_times(
            duration=meta.duration_sec,
            clip_dur=clip_dur,
            event_times=[e.time for e in ann.events],
            n=neg_per_video,
            min_gap=max(neg_min_gap, clip_dur / 2.0),
            rng=rng,
        )
        centers += [("background", t) for t in neg_times]

        desc = f"{video_path.name}: {len(events)} pos, {len(neg_times)} neg"
        for k, (label, center) in enumerate(tqdm(centers, desc=desc, leave=False)):
            try:
                # Sample full-resolution frames, then apply the same HUD mask +
                # spatial fit the recognizer/live capture use, so train == serve.
                clip = video_utils.sample_clip(
                    video_path,
                    center_sec=center,
                    num_frames=cfg.num_frames,
                    sample_fps=cfg.sample_fps,
                    resize_short=None,
                )
                clip = video_utils.preprocess_clip(
                    clip, cfg.crop_size, hud_mask=cfg.hud_mask, frame_mode=cfg.frame_mode)
            except Exception as exc:  # noqa: BLE001 - keep going on a bad clip
                print(f"  ! failed clip @ {center:.2f}s in {video_path.name}: {exc}")
                continue
            clip_name = f"{video_path.stem}__{label}__{k:04d}.npy"
            clip_path = frames_dir / clip_name
            np.save(clip_path, clip)
            items.append({
                "clip": str(clip_path.relative_to(out_dir)),
                "label": label,
                "label_id": cfg.class_to_id()[label],
                "video": video_path.name,
                "time": round(float(center), 4),
            })

    if not items:
        raise SystemExit("No clips were produced. Check that videos and annotations line up.")

    # --- stratified train/val split ---
    val_fraction = float(ds.get("val_fraction", 0.2))
    by_label: Dict[str, List[int]] = defaultdict(list)
    for idx, it in enumerate(items):
        by_label[it["label"]].append(idx)
    for label, idxs in by_label.items():
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * val_fraction))
        val_set = set(idxs[:n_val])
        for idx in idxs:
            items[idx]["split"] = "val" if idx in val_set else "train"

    manifest = {
        "champion": cfg.champion,
        "classes": cfg.classes,
        "clip": {
            "num_frames": cfg.num_frames,
            "sample_fps": cfg.sample_fps,
            "crop_size": cfg.crop_size,
            "frame_mode": cfg.frame_mode,
            "hud_mask": cfg.hud_mask,
        },
        "items": items,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # --- report ---
    counts = Counter((it["label"], it["split"]) for it in items)
    print("\nBuilt clips:")
    for label in cfg.classes:
        tr = counts.get((label, "train"), 0)
        va = counts.get((label, "val"), 0)
        print(f"  {label:<12} train={tr:<5} val={va:<5} total={tr + va}")
    if skipped:
        print(f"  ({skipped} events skipped: ability not in config classes)")
    print(f"\nManifest -> {manifest_path}")
    print(f"Clips    -> {frames_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build training clips from timestamp annotations.")
    p.add_argument("--config", required=True)
    p.add_argument("--raw-dir", default="data/raw_videos")
    p.add_argument("--annotations-dir", default="data/annotations")
    p.add_argument("--clips-root", default="data/clips")
    args = p.parse_args()
    build(args.config, args.raw_dir, args.annotations_dir, args.clips_root)


if __name__ == "__main__":
    main()
