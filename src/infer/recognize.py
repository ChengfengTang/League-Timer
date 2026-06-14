"""Sliding-window ability recognizer over a recorded video file.

Loads a trained checkpoint, slides a fixed-length window across the video,
classifies each window, then converts the per-window scores into discrete
ability events via per-class thresholding + temporal non-max suppression.

Run::

    python -m src.infer.recognize --config configs/ezreal.yaml \
        --video data/raw_videos/ezreal_test.mp4 \
        --checkpoint models/ezreal/best.pt --overlay

Outputs:
  - outputs/<stem>.events.json : [{"ability", "time", "score"}, ...]
  - outputs/<stem>.overlay.mp4 : annotated video (only with --overlay)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from src.common import video as vu
from src.common.config import Config
from src.common.torch_utils import pick_device
from src.dataset.clip_dataset import ClipTransform
from src.train.model import build_model


def load_model(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, _spec, _head = build_model(ckpt["backbone"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def sliding_window_scores(
    video_path: str,
    model,
    ckpt: Dict,
    device: torch.device,
    stride_sec: float,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (times, probs) where probs has shape (num_windows, num_classes)."""
    meta = vu.probe(video_path)
    num_frames = int(ckpt["num_frames"])
    sample_fps = float(ckpt["sample_fps"])
    crop_size = int(ckpt["crop_size"])
    clip_dur = num_frames / sample_fps

    transform = ClipTransform(crop_size, ckpt["mean"], ckpt["std"], train=False)

    lo = clip_dur / 2.0
    hi = max(meta.duration_sec - clip_dur / 2.0, lo)
    centers = list(np.arange(lo, hi + 1e-6, stride_sec)) or [meta.duration_sec / 2.0]

    times: List[float] = []
    probs: List[np.ndarray] = []
    batch: List[torch.Tensor] = []
    batch_times: List[float] = []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.extend(list(p))
        times.extend(batch_times)
        batch.clear()
        batch_times.clear()

    from tqdm import tqdm
    for c in tqdm(centers, desc="scanning"):
        clip = vu.sample_clip(video_path, center_sec=float(c), num_frames=num_frames,
                              sample_fps=sample_fps, resize_short=crop_size)
        batch.append(transform(clip))
        batch_times.append(float(c))
        if len(batch) >= batch_size:
            flush()
    flush()

    order = np.argsort(times)
    return np.array(times)[order], np.array(probs)[order]


def detect_events(
    times: np.ndarray,
    probs: np.ndarray,
    classes: List[str],
    thresholds: Dict[str, float],
    default_threshold: float,
    nms_window_sec: float,
) -> List[Dict]:
    """Threshold + greedy temporal NMS, per ability class."""
    events: List[Dict] = []
    for ci, name in enumerate(classes):
        if name == "background":
            continue
        thr = float(thresholds.get(name, default_threshold))
        scores = probs[:, ci]
        cand = [i for i in range(len(times)) if scores[i] >= thr]
        cand.sort(key=lambda i: scores[i], reverse=True)
        taken_times: List[float] = []
        for i in cand:
            t = float(times[i])
            if all(abs(t - tt) >= nms_window_sec for tt in taken_times):
                taken_times.append(t)
                events.append({"ability": name, "time": round(t, 3),
                               "score": round(float(scores[i]), 4)})
    events.sort(key=lambda e: e["time"])
    return events


def _nearest_idx(times: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(times - t)))


def render_overlay(
    video_path: str,
    out_path: Path,
    times: np.ndarray,
    probs: np.ndarray,
    events: List[Dict],
    classes: List[str],
    flash_sec: float = 0.6,
) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    ability_idx = [i for i, c in enumerate(classes) if c != "background"]
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = f / fps
        # live top-1 ability readout from nearest scored window
        if len(times):
            j = _nearest_idx(times, t)
            best_i = max(ability_idx, key=lambda i: probs[j, i])
            readout = f"{classes[best_i]} {probs[j, best_i]:.2f}"
        else:
            readout = "-"
        cv2.putText(frame, readout, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, readout, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        # flash detected events
        active = [e for e in events if 0 <= (t - e["time"]) <= flash_sec]
        for k, e in enumerate(active):
            label = f"CAST {e['ability']} ({e['score']:.2f})"
            y = 70 + 34 * k
            cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 80, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        f += 1
    cap.release()
    writer.release()


def run(config_path: str, video_path: str, checkpoint: str, out_dir: str,
        overlay: bool, device_str: str) -> None:
    cfg = Config.load(config_path)
    icfg = cfg.section("infer")
    device = pick_device(device_str)
    print(f"Device: {device}")

    model, ckpt = load_model(checkpoint, device)
    classes = ckpt["classes"]
    print(f"Loaded {ckpt['backbone']} ({ckpt.get('champion', '?')}), classes={classes}")

    times, probs = sliding_window_scores(
        video_path, model, ckpt, device,
        stride_sec=float(icfg.get("stride_sec", 0.25)),
    )
    events = detect_events(
        times, probs, classes,
        thresholds=icfg.get("thresholds", {}) or {},
        default_threshold=float(icfg.get("default_threshold", 0.6)),
        nms_window_sec=float(icfg.get("nms_window_sec", 1.0)),
    )

    stem = Path(video_path).stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / f"{stem}.events.json"
    with open(events_path, "w") as f:
        json.dump({"video": Path(video_path).name, "events": events}, f, indent=2)
    print(f"\nDetected {len(events)} events -> {events_path}")
    for e in events:
        print(f"  {e['time']:8.2f}s  {e['ability']:<4} ({e['score']:.2f})")

    if overlay:
        overlay_path = out_dir / f"{stem}.overlay.mp4"
        print(f"\nRendering overlay -> {overlay_path}")
        render_overlay(video_path, overlay_path, times, probs, events, classes)
        print("Overlay done.")


def main() -> None:
    p = argparse.ArgumentParser(description="Sliding-window ability recognizer.")
    p.add_argument("--config", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--overlay", action="store_true", help="also render an annotated video")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    args = p.parse_args()
    run(args.config, args.video, args.checkpoint, args.out_dir, args.overlay, args.device)


if __name__ == "__main__":
    main()
