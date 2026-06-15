"""Overlay localization + level-read ROI on a video for tuning.

Draws:
  - red   = healthbar
  - green = champion crop (classifier input)
  - cyan  = level-read ROI (badge left of bar)

Run::

    python -m src.localize.preview_level --config configs/{ChampionName}.yaml \\
        --video data/{ChampionName}/raw_videos/test.mp4 --out outputs/level_preview.mp4

Still frames with the ROI crop are also saved under ``outputs/level_roi_samples/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.common.config import Config
from src.localize import Localizer
from src.localize.level_reader import LevelReader


def run(config_path: str, video: str, out: str, start: float, end: float | None,
        stride: int, save_samples: int) -> None:
    cfg = Config.load(config_path)
    loc = Localizer.from_config(cfg.section("localize"), base_dir=".")
    lr_cfg = cfg.section("localize").get("level_read") or {}
    if not lr_cfg.get("enabled"):
        print("Warning: localize.level_read.enabled is false in config")
    reader = LevelReader.from_config(lr_cfg, base_dir=".")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video {w}x{h} @ {fps:.1f} fps")
    print(
        "Level ROI relative to healthbar top-left: "
        f"offset_x={reader.offset_x}, offset_y={reader.offset_y}, "
        f"size={reader.width}x{reader.height}"
    )

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sample_dir = Path("outputs/level_roi_samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h))

    start_f = int(round(start * fps))
    end_f = int(round(end * fps)) if end is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    idx, hits, level_hits, saved = start_f, 0, 0, 0
    while idx < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start_f) % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = loc.locate(rgb)
            total_frame = True
            hits += 1 if dets else 0
            for d in dets:
                bx, by, bw, bh = d.bar
                x, y, cw, ch = d.box
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{d.champion} {d.score:.2f}", (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

                dbg = reader.read_debug(rgb, d.bar)
                lbox = dbg.get("box")
                if lbox is not None:
                    lx, ly, lw, lh = lbox
                    cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), (255, 255, 0), 2)
                    label = "lvl ?"
                    if dbg.get("level") is not None:
                        level_hits += 1
                        label = f"lvl {dbg['level']} ({dbg['confidence']:.2f})"
                    else:
                        scores = dbg.get("scores") or {}
                        if scores:
                            best = max(scores, key=scores.get)
                            label = f"lvl ? best={best}@{scores[best]:.2f}"
                    cv2.putText(
                        frame, label, (lx, max(12, ly - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2,
                    )
                    if saved < save_samples:
                        roi = reader._level_roi(rgb, d.bar)
                        if roi is not None:
                            cv2.imwrite(
                                str(sample_dir / f"frame{idx:06d}_roi.png"),
                                cv2.cvtColor(roi, cv2.COLOR_RGB2BGR),
                            )
                            saved += 1
            writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    processed = max(1, (idx - start_f) // stride)
    print(f"Wrote {out}")
    print(f"  localized {hits}/{processed} frames")
    print(f"  level read OK on {level_hits} detection(s)")
    if saved:
        print(f"  saved {saved} ROI crops -> {sample_dir}/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="outputs/level_preview.mp4")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--save-samples", type=int, default=5,
                   help="save this many ROI crop PNGs for inspection")
    args = p.parse_args()
    run(args.config, args.video, args.out, args.start, args.end, args.stride,
        args.save_samples)


if __name__ == "__main__":
    main()
