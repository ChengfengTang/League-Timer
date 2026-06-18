"""Combined preview: ability casts + localization boxes + level read.

Draws on every frame:
  - red    = healthbar
  - green  = champion crop (classifier input)
  - blue   = level-read ROI
  - top    = model top-1 readout + CAST flashes (from sliding-window inference)

Run::

    python -m src.infer.preview_combined --config configs/ezreal.yaml \\
        --checkpoint models/ezreal/best.pt \\
        --video data/ezreal/raw_videos/ezreal1.mp4
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np

from src.common.config import Config
from src.common.torch_utils import pick_device
from src.infer.recognize import (
    _nearest_idx,
    _put_centered_text,
    detect_events,
    load_model,
    sliding_window_scores,
)
from src.localize import Localizer
from src.localize.level_reader import LevelReader
from src.timers import RBarDetector, load_r_bar, merge_events, scan_video_events

# BGR
COLOR_BAR = (0, 0, 255)
COLOR_CROP = (0, 255, 0)
COLOR_LEVEL = (255, 0, 0)
COLOR_RBAR = (255, 0, 255)


def render_combined(
    video_path: str,
    out_path: Path,
    times: np.ndarray,
    probs: np.ndarray,
    events: List[Dict],
    classes: List[str],
    loc: Localizer,
    reader: LevelReader | None,
    r_bar: RBarDetector | None = None,
    flash_sec: float = 0.6,
) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    ability_idx = [i for i, c in enumerate(classes) if c != "background"]
    f = loc_hits = level_hits = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = f / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = loc.locate(rgb)
        if dets:
            loc_hits += 1
            d = max(dets, key=lambda x: (x.score, x.box[2] * x.box[3]))
            bx, by, bw, bh = d.bar
            x, y, cw, ch = d.box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), COLOR_BAR, 2)
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), COLOR_CROP, 2)
            cv2.putText(
                frame, f"{d.champion} {d.score:.2f}", (x, max(14, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_CROP, 2, cv2.LINE_AA,
            )
            if r_bar is not None:
                _, r_roi, r_present = r_bar.score(rgb, d.bar)
                rx, ry, rw, rh = r_roi
                cv2.rectangle(
                    frame, (rx, ry), (rx + rw, ry + rh),
                    COLOR_RBAR if r_present else (128, 0, 128), 2,
                )
            if reader is not None:
                dbg = reader.read_debug(rgb, d.bar)
                lbox = dbg.get("box")
                if lbox is not None:
                    lx, ly, lw, lh = lbox
                    cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), COLOR_LEVEL, 2)
                    if dbg.get("level") is not None:
                        level_hits += 1
                        lbl = f"lvl {dbg['level']} ({dbg['confidence']:.2f})"
                    else:
                        scores = dbg.get("scores") or {}
                        best = max(scores, key=scores.get) if scores else "?"
                        lbl = f"lvl ? {best}@{scores.get(best, 0):.2f}" if scores else "lvl ?"
                    cv2.putText(
                        frame, lbl, (lx, max(14, ly - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LEVEL, 2, cv2.LINE_AA,
                    )

        cx = w // 2
        if len(times):
            j = _nearest_idx(times, t)
            best_i = max(ability_idx, key=lambda i: probs[j, i])
            readout = f"{classes[best_i]} {probs[j, best_i]:.2f}"
        else:
            readout = "-"
        _put_centered_text(frame, readout, cx, 56, 2.0, (0, 255, 255), 3, outline_thick=7)

        active = [e for e in events if 0 <= (t - e["time"]) <= flash_sec]
        for k, e in enumerate(active):
            y = 100 + 46 * k
            if y > h - 36:
                break
            _put_centered_text(
                frame, f"CAST {e['ability']} ({e['score']:.2f})", cx, y,
                1.4, (0, 80, 255), 3, outline_thick=6,
            )

        legend = "red=bar  green=crop  blue=level  magenta=r_bar"
        cv2.putText(
            frame, legend, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (220, 220, 220), 1, cv2.LINE_AA,
        )
        writer.write(frame)
        f += 1

    cap.release()
    writer.release()
    print(f"  overlay: {loc_hits}/{f} frames localized, {level_hits} with level read")


def run(
    config_path: str,
    video_path: str,
    checkpoint: str,
    out_dir: str,
    device_str: str,
) -> None:
    cfg = Config.load(config_path)
    icfg = cfg.section("infer")
    device = pick_device(device_str)
    print(f"Device: {device}")

    model, ckpt = load_model(checkpoint, device)
    classes = ckpt["classes"]
    print(f"Loaded {ckpt['backbone']} ({ckpt.get('champion', '?')}), classes={classes}")

    print("Running sliding-window inference...")
    times, probs = sliding_window_scores(
        video_path, model, ckpt, device,
        stride_sec=float(icfg.get("stride_sec", 0.25)),
    )
    events = detect_events(
        times, probs, classes,
        thresholds=icfg.get("thresholds", {}) or {},
        default_threshold=float(icfg.get("default_threshold", 0.6)),
        nms_window_sec=float(icfg.get("nms_window_sec", 1.0)),
        peak_only=bool(icfg.get("peak_only", False)),
        min_margin=float(icfg.get("min_margin", 0.0)),
        track=icfg.get("track") or None,
    )

    loc = Localizer.from_config(cfg.section("localize"), base_dir=".")
    r_bar = load_r_bar(cfg, base_dir=".")
    if r_bar is not None:
        r_bar_cfg = cfg.section("timers").get("r_bar") or {}
        ability = str(r_bar_cfg.get("ability", "R"))
        stride = int(r_bar_cfg.get("scan_stride_frames", 1))
        r_events = scan_video_events(
            video_path, loc, r_bar, ability=ability, stride_frames=stride,
        )
        if r_events:
            print(f"  (+{len(r_events)} {ability} from r_bar icon detector)")
        events = merge_events(events, r_events)

    stem = Path(video_path).stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / f"{stem}.events.json"
    with open(events_path, "w") as f:
        json.dump({"video": Path(video_path).name, "events": events}, f, indent=2)
    print(f"\n{stem}: {len(events)} events -> {events_path}")
    for e in events:
        print(f"  {e['time']:8.2f}s  {e['ability']:<5} ({e['score']:.2f})")

    lr_cfg = cfg.section("localize").get("level_read") or {}
    reader = (
        LevelReader.from_config(lr_cfg, base_dir=".")
        if lr_cfg.get("enabled")
        else None
    )

    overlay_path = out_dir / f"{stem}.combined.mp4"
    print(f"Rendering {overlay_path} ...")
    render_combined(
        video_path, overlay_path, times, probs, events, classes, loc, reader,
        r_bar=r_bar,
    )
    print(f"Done -> {overlay_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    run(args.config, args.video, args.checkpoint, args.out_dir, args.device)


if __name__ == "__main__":
    main()
