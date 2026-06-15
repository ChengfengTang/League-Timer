"""Overlay champion localization on a video to eyeball the localizer.

Draws each detection's healthbar box (red) and champion crop box (green, the
region the classifier will see) onto every frame and writes an output video.

Run::

    python -m src.localize.preview --config configs/{ChampionName}.yaml \
        --video data/{ChampionName}/raw_videos/test.mp4 --out outputs/loc_test.mp4

Tip: real games have several enemies, so set ``localize.assume_single_enemy:
false`` and provide ``name_templates`` to keep only the tracked champion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.common.config import Config
from src.localize import Localizer


def run(config_path: str, video: str, out: str, start: float, end: float | None,
        stride: int) -> None:
    cfg = Config.load(config_path)
    loc = Localizer.from_config(cfg.section("localize"), base_dir=".")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h))

    start_f = int(round(start * fps))
    end_f = int(round(end * fps)) if end is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    idx, hits, total = start_f, 0, 0
    while idx < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start_f) % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = loc.locate(rgb)
            total += 1
            hits += 1 if dets else 0
            for d in dets:
                bx, by, bw, bh = d.bar
                x, y, cw, ch = d.box
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
                cv2.putText(frame, f"{d.champion} {d.score:.2f}", (x, max(12, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    rate = (hits / total) if total else 0.0
    print(f"Wrote {out}  ({total} frames, detection on {hits} = {rate:.1%})")


def main() -> None:
    p = argparse.ArgumentParser(description="Preview champion localization on a video.")
    p.add_argument("--config", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=float, default=0.0, help="start time (sec)")
    p.add_argument("--end", type=float, default=None, help="end time (sec)")
    p.add_argument("--stride", type=int, default=2, help="process every Nth frame")
    args = p.parse_args()
    run(args.config, args.video, args.out, args.start, args.end, args.stride)


if __name__ == "__main__":
    main()
