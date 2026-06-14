# LoL Ability VFX Detector

Detect a League of Legends champion's spell VFX (Q/W/E/R) from recorded gameplay
video. The pipeline is config-driven: you train one champion at a time by
recording footage, labelling ability casts by timestamp, fine-tuning a small
video classifier (in the cloud), and then running a sliding-window recognizer
over new video to emit timestamped ability events.

> Scope of the MVP: one champion, all four abilities, offline recognition over
> recorded video files. Live screen capture and multi-champion-on-screen are
> deliberate non-goals (see "Roadmap").

## Pipeline at a glance

```
record video -> annotate (timestamps) -> build clips -> train (cloud) -> recognize -> evaluate
```

Each stage reads the same per-champion config (`configs/<champion>.yaml`), so the
clip length / fps / classes stay consistent end to end.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists `torch`/`torchvision` generically. Pick the build for
your platform:

- Mac (Apple Silicon, for inference): `pip install torch torchvision`
- CUDA cloud GPU (for training): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

## 1. Record gameplay

Put recordings in `data/raw_videos/`. Guidelines that make labelling and training
much easier:

- **One champion per recording session** (the champion you're training).
- **Consistent resolution and fps** across recordings; 1080p @ 30-60 fps is fine.
  Higher fps captures fast VFX better but means more frames to read.
- **Cast every ability many times** in varied situations: different targets,
  different map locations, with/without minions and other champions around,
  different times of day in-game, and ideally a couple of skins.
- Aim for roughly **50-150 clean casts per ability** before expecting the model
  to generalize. Short, dense clips (lots of casting) are better than long idle
  footage.
- Keep some footage aside, unlabelled-for-training, as a **held-out test video**.

## 2. Annotate (timestamp labels)

Scrub a video and press a key at the moment each ability is cast:

```bash
python -m src.annotate.annotate --config configs/ezreal.yaml \
    --video data/raw_videos/ezreal_game1.mp4
```

Controls are printed on launch (play/pause, frame step, seek, press `Q/W/E/R` to
log a cast, undo, delete-near, save). Output goes to
`data/annotations/<video_stem>.json`. Re-running on the same video resumes from
the existing annotations.

## 3. Build the clip dataset

Turn timestamp events into fixed-length training clips (positives) plus sampled
`background` windows (negatives), with a train/val split:

```bash
python -m src.dataset.build --config configs/ezreal.yaml
```

Writes clips under `data/clips/<champion>/` and a `manifest.json` describing the
split. Clips are stored as `.npy` frame stacks so training doesn't re-decode
video.

## 4. Train

Train locally or on a cloud GPU (CUDA recommended):

```bash
python -m src.train.train --config configs/ezreal.yaml
```

Checkpoints land in `models/<champion>/best.pt`. Training prints per-class
precision/recall and a confusion matrix on the validation split.

## 5. Recognize (sliding window over a video)

```bash
python -m src.infer.recognize --config configs/ezreal.yaml \
    --video data/raw_videos/ezreal_test.mp4 \
    --checkpoint models/ezreal/best.pt \
    --overlay
```

Outputs `outputs/<video_stem>.events.json` (list of `{ability, time, score}`)
and, with `--overlay`, an annotated `.mp4` drawing detections for eyeballing.

## 6. Evaluate

Score predicted events against hand labels with a time tolerance:

```bash
python -m src.infer.evaluate --config configs/ezreal.yaml \
    --pred outputs/ezreal_test.events.json \
    --truth data/annotations/ezreal_test.json
```

Prints event-level precision / recall / F1 per ability. Use the misses to decide
what extra footage to record.

## Adding a new champion

Copy `configs/ezreal.yaml` to `configs/<champion>.yaml`, adjust `champion` and
(if the kit needs it) the class list and thresholds, then repeat steps 1-6.

## Repo layout

```
configs/        per-champion YAML configs
data/
  raw_videos/   recorded gameplay (git-ignored)
  annotations/  timestamp label JSON (git-ignored)
  clips/        extracted training clips + manifest (git-ignored)
src/
  common/       config, video sampling, annotation schema, torch helpers
  annotate/     timestamp annotation tool
  dataset/      clip extraction, train/val split, PyTorch clip dataset
  train/        model + training loop
  infer/        sliding-window recognizer + evaluation
models/         trained checkpoints (git-ignored; see below)
outputs/        predicted events + overlay videos (git-ignored)
AGENTS.md       detection tuning notes (recall > precision)
```

`models/` is git-ignored for the same reason as `data/clips/`: checkpoints are
large binary artifacts (~10 MB each) produced by training, not source. They
change every retrain and are easy to regenerate with `src.train.train`. Only
`models/.gitkeep` is tracked so the directory exists after clone.

## Roadmap (post-MVP)

- Live recognition from screen capture (point the recognizer at a capture source).
- Multiple champions on screen (add localization/cropping before classification).
- Per-champion cropping driven by the minimap / HUD to reduce background noise.
