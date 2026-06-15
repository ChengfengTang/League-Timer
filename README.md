# LoL Ability VFX Detector

Detect a League of Legends champion's spell VFX (Q/W/E/R) from recorded gameplay
video. The pipeline is config-driven: you train one champion at a time by
recording footage, labelling ability casts by timestamp, fine-tuning a small
video classifier (in the cloud), and then running a sliding-window recognizer
over new video to emit timestamped ability events.

> Scope of the MVP: one champion, all four abilities, offline recognition over
> recorded video files. Live screen capture and multi-champion-on-screen are
> deliberate non-goals (see "Roadmap").

[![Watch Video Demo]](https://drive.google.com/file/d/1iFoqqnDQptAq1k3ml819wMywCJUmYQx-/view?usp=sharing)
## Pipeline at a glance

```
record video -> annotate (timestamps) -> build clips -> train (cloud) -> recognize -> evaluate
```

Each stage reads the same per-champion config (`configs/{ChampionName}.yaml`), so the
clip length / fps / classes stay consistent end to end.

For a higher-level engineering write-up of the model, image-processing pipeline,
and the full-screen to localize-then-classify evolution, see
[`docs/technical-writeup.md`](docs/technical-writeup.md).

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

Put recordings in `data/{ChampionName}/raw_videos/`. Guidelines that make labelling and training
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
python -m src.annotate.annotate --config configs/{ChampionName}.yaml \
    --video data/{ChampionName}/raw_videos/game1.mp4
```

Controls are printed on launch (play/pause, frame step, seek, press `Q/W/E/R` to
log a cast, undo, delete-near, save). Output goes to
`data/{ChampionName}/annotations/<video_stem>.json`. Re-running on the same video resumes from
the existing annotations.

## 3. Build the clip dataset

Turn timestamp events into fixed-length training clips (positives) plus sampled
`background` windows (negatives), with a train/val split:

```bash
python -m src.dataset.build --config configs/{ChampionName}.yaml
```

Writes clips under `data/{ChampionName}/clips/` and a `manifest.json` describing the
split. Clips are stored as `.npy` frame stacks so training doesn't re-decode
video.

### Frame preprocessing (`clip.frame_mode` / `clip.hud_mask`)

How each frame is fit to the square model input is config-driven and applied
identically by the builder, recognizer, and live capture (so train == serve):

- `frame_mode: letterbox` (default) keeps the **whole frame** (aspect-preserving
  resize + pad). Use this to detect casts **anywhere on screen** (e.g. an enemy
  champion near the edges). `center_crop` is the legacy mode that resizes the
  short side then center-crops, dropping the left/right edges — only safe when
  the thing you track is always centered (your own champion).
- `hud_mask` blacks out rectangles (given as `[x, y, w, h]` fractions) before the
  frame is used. This hides your own HUD (ability bar / minimap) so the model
  can't cheat off your cooldown sweep or mana flash — cues that don't exist for
  an enemy caster. Tune the defaults to your resolution/layout; set `[]` to
  disable.
- `spatial_jitter` (train-time only) randomly zooms/repositions clips so casts
  are learned at many on-screen locations and scales.

Changing any of these means you must **rebuild** (`src.dataset.build`) before
retraining, since stored clips are baked at build time.

### Localization (`localize:`) — find the champion before classifying

Full-frame classification breaks down in real games: the champion you track is
small, off-centre, and surrounded by *other* champions' abilities (any stray
Flash becomes noise). When `localize.enabled: true`, the builder/recognizer
first **finds the tracked champion** and classifies a tight crop around it
instead of the whole frame. This removes teamfight noise and enlarges small VFX.

How it finds the champion (`src/localize/`):

- **Name + healthbar** — identity is the champion name above the bar (e.g.
  "Ezreal"), not bar colour. Allies use a **green** bar; enemies use **red**;
  either works the same. Template-match the name, estimate the bar just below
  it, and crop a `box_size` square under the bar. Add more champions by adding
  more `name_templates`.
- **Bar-colour fallback** — if name matching fails, optional red/green HSV bar
  detection finds champion-shaped bars and confirms them by name. Minion bars
  are filtered by shape (wide, thin).
- **`ignore_regions`** — fractional `[x, y, w, h]` UI zones to skip (HUD,
  scoreboard, minimap). Tune to your layout.

The template must come from **your client language** (e.g. English "Ezreal").
The interface returns a list of `(champion, box)` for multi-champion tracking
(enemies or teammates).

Preview the localizer on any video (red outline = estimated bar, green = crop):

```bash
python -m src.localize.preview --config configs/{ChampionName}.yaml \
    --video data/{ChampionName}/raw_videos/test.mp4 --out outputs/loc_test.mp4
```

Like the preprocessing keys, changing `localize:` means you must **rebuild**.

## 4. Train

Train locally or on a cloud GPU (CUDA recommended):

```bash
python -m src.train.train --config configs/{ChampionName}.yaml
```

Checkpoints land in `models/{ChampionName}/best.pt`. Training prints per-class
precision/recall and a confusion matrix on the validation split.

## 5. Recognize (sliding window over a video)

```bash
python -m src.infer.recognize --config configs/{ChampionName}.yaml \
    --video data/{ChampionName}/raw_videos/test.mp4 \
    --checkpoint models/{ChampionName}/best.pt \
    --overlay
```

Outputs `outputs/<video_stem>.events.json` (list of `{ability, time, score}`)
and, with `--overlay`, an annotated `.mp4` drawing detections for eyeballing.

## 6. Evaluate

Score predicted events against hand labels with a time tolerance:

```bash
python -m src.infer.evaluate --config configs/{ChampionName}.yaml \
    --pred outputs/test.events.json \
    --truth data/{ChampionName}/annotations/test.json
```

Prints event-level precision / recall / F1 per ability. Use the misses to decide
what extra footage to record.

## Live (real-time screen capture)

Watch your screen *as you play* and print `CAST W/E/R/Flash` the moment an
ability fires (same model + thresholds as the offline recognizer):

```bash
python -m src.infer.live --config configs/{ChampionName}.yaml \
    --checkpoint models/{ChampionName}/best.pt
```

A background thread grabs frames into a rolling one-clip buffer at the model's
`sample_fps`; the main loop classifies the latest clip every `infer.stride_sec`
and a streaming detector applies the same threshold / `min_margin` / `peak_only`
/ `nms_window_sec` logic before emitting an event.

Useful flags:

- `--monitor 1` capture a specific display (`1` = primary, `0` = all combined).
- `--region x,y,w,h` capture an explicit rectangle instead of a whole monitor.
- `--stride-sec 0.2` override how often inference runs (lower = snappier, heavier).
- `--device mps|cpu|cuda` force a device (`auto` by default).
- `--verbose` show a live top-1 readout + capture fps.
- `--preview` open a window showing your screen with the `clip.hud_mask` boxes drawn
  next to the exact masked/letterboxed frame the model sees. Use it to align the
  mask to your HUD before recording/training (it reflects the config, not the
  checkpoint). Press `q` in the window to stop.
- `--countdown 5` seconds to alt-tab into the game before it starts.

> **macOS:** the first run triggers a Screen Recording permission prompt
> (System Settings -> Privacy & Security -> Screen Recording). Grant it to your
> terminal / IDE, then restart the command. Capture the same resolution/zoom you
> trained on for best results; keep the game on the captured monitor.

## Web app (timer UI + auto-detection)

A local web app that tracks ability and summoner-spell cooldowns. Add a
champion and the model auto-starts timers when it detects a cast; click any
ability/summoner to start it manually.

```bash
python -m src.app.server      # then open http://127.0.0.1:8000
```

How it works:

- One Python process serves the static UI (`src/app/static/`) and a `/ws`
  WebSocket that streams timer state ~4x/sec.
- Adding a champion that has both `configs/{ChampionName}.yaml` and
  `models/{ChampionName}/best.pt` starts a `LiveDetector`
  (`src/app/detector.py`, the same tuned capture/inference loop as
  `src.infer.live`). Detected `Flash`/`W`/`E`/`R` casts auto-start that
  champion's timers.
- Champions without a model are **manual-only** — the card shows only tracked
  abilities/summoners from the config (`infer.track` + `timers.summoners`).
- Ability base CDs live in `configs/{ChampionName}.yaml` (`timers.abilities`).
  Summoner base CDs use a built-in table in `src/app/cooldowns.py` (Flash 300,
  Ignite 180, etc.). Ability haste, summoner haste, and per-rank scaling are
  applied when a timer starts.

### Sync ability cooldowns from Data Dragon

Ability CDs are **not** fetched at runtime — they are stored in the champion
config and must be refreshed after Riot patches. The sync script pulls the
current patch from [Data Dragon](https://developer.riotgames.com/docs/lol#data-dragon)
and updates `timers.abilities` for abilities listed in `infer.track` (skips Q
and summoners):

```bash
python scripts/sync_timer_cds.py {ChampionName}
```

Pin a specific patch if needed:

```bash
python scripts/sync_timer_cds.py {ChampionName} --version 16.12.1
```

Restart the web server after syncing so it picks up the new yaml.

Config (`configs/{ChampionName}.yaml`):

```yaml
timers:
  abilities:        # base CDs from Data Dragon — run scripts/sync_timer_cds.py
    W: 8
    E: [26, 23, 20, 17, 14]
    R: [120, 105, 90]
  summoners: [Flash]   # summoners this champion's model can auto-detect
  class_to_key: {}     # optional model-class -> timer-key remap (identity if empty)
```

Env overrides: `LEAGUE_TIMER_HOST`, `LEAGUE_TIMER_PORT` (default `127.0.0.1:8000`),
`LEAGUE_TIMER_DEVICE` (`auto`|`cpu`|`mps`|`cuda`).

> Same macOS Screen Recording note as Live applies — the prompt appears when you
> add the first modeled champion (that's when capture begins).
>
> Scope: one auto-detector runs at a time (single screen, ~230 ms/inference); a
> second modeled champion added while one is running stays manual-only. No voice
> (input or parsing) — manual tracking is click-only.

## Adding a new champion

Copy an existing champion config to `configs/{ChampionName}.yaml`, adjust `champion` and
(if the kit needs it) the class list and thresholds, then repeat steps 1-6. Seed
ability CDs from Data Dragon:

```bash
python scripts/sync_timer_cds.py {ChampionName}
```

## Repo layout

```
configs/        per-champion YAML configs
scripts/        utilities (e.g. sync_timer_cds.py — refresh timers from DDRagon)
data/
  {ChampionName}/  per-champion data (git-ignored)
    raw_videos/  recorded gameplay
    annotations/ timestamp label JSON
    clips/       extracted training clips + manifest
src/
  common/       config, video sampling, annotation schema, torch helpers
  annotate/     timestamp annotation tool
  dataset/      clip extraction, train/val split, PyTorch clip dataset
  localize/     champion localization (healthbar + name templates) + preview
  train/        model + training loop
  infer/        sliding-window recognizer + evaluation + live screen capture
  app/          web app: FastAPI server, cooldown engine, LiveDetector, static UI
models/         trained checkpoints (git-ignored; see below)
outputs/        predicted events + overlay videos (git-ignored)
AGENTS.md       detection tuning notes (recall > precision)
```

`models/` is git-ignored for the same reason as `data/{ChampionName}/clips/`: checkpoints are
large binary artifacts (~10 MB each) produced by training, not source. They
change every retrain and are easy to regenerate with `src.train.train`. Only
`models/.gitkeep` is tracked so the directory exists after clone.

## Roadmap (post-MVP)

- ~~Live recognition from screen capture~~ — done, see `src.infer.live`.
- ~~Cooldown timer UI driven by the model~~ — done, see `src.app.server`.
- ~~Per-champion cropping to reduce background noise~~ — done, see `localize:`
  and `src/localize/` (healthbar localization + champion-centred crops).
- Web app: multiple concurrent auto-detectors and per-champion summoner-spell presets.
- Multiple champions on screen: localizer interface is multi-champion ready;
  wire `name_templates` + flip `assume_single_enemy: false` and integrate the
  crop step into `recognize.py` / `live.py`.
- Flash disambiguation via the localized box displacement (spatial cue).
