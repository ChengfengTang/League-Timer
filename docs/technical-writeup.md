# Technical Write-Up: Ezreal VFX Detection and Cooldown Tracking

This document is a short engineering record of the current League Timer system:
what we built, what changed during development, and why the final approach uses
both model training and classical image processing. It is intentionally more
detailed than the README, but less formal than a paper.

## Project Goal

The goal is to detect enemy League of Legends ability casts from gameplay video
or live screen capture, then start cooldown timers automatically. The current
champion target is Ezreal, with tracked outputs for high-value cooldowns:
`W`, `E`, `R`, and `Flash`.

The important product constraint is that missing a real cast is worse than
showing an occasional extra timer. A false positive is annoying; a false
negative makes the user believe an ability is available when it is actually on
cooldown. That means the inference pipeline is tuned for recall first, then
precision is improved with filters such as margins, local peaks, and temporal
non-max suppression.

## Methods & Technologies Used

Quick reference of the main techniques in this repo. Useful if you want to talk
about the project in an interview or portfolio review.

### Computer vision and image processing

| Topic | What we used it for |
|-------|---------------------|
| **OpenCV** | Video read/write, annotation UI, overlay rendering, live preview |
| **HSV color segmentation** | Detect red/green champion healthbars (`cv2.inRange`, hue wraps for red) |
| **Morphological ops** | Close gaps in bar masks (`cv2.morphologyEx`) |
| **Contour detection** | Find bar-shaped blobs, filter by width/height/aspect ratio |
| **Template matching** | Match champion nameplate text (`cv2.matchTemplate`, multi-scale) |
| **Letterboxing** | Fit full gameplay frame into square model input without cropping edges |
| **HUD masking** | Black out ability bar / minimap so the model cannot cheat on UI cues |
| **Spatial jitter** | Train-time random zoom + crop for location/scale robustness |

### Deep learning and training

| Topic | What we used it for |
|-------|---------------------|
| **PyTorch** | Training loop, inference, checkpoint save/load |
| **X3D-S** | Main video classifier backbone (3D CNN, pretrained on Kinetics via PyTorchVideo) |
| **Transfer learning** | Replace the final layer, fine-tune on Ezreal clips |
| **Head warm-up** | Freeze backbone for first N epochs, train classifier head only |
| **Cross-entropy loss** | Multi-class clip classification |
| **Class weighting** | Inverse-frequency weights so rare abilities are not ignored |
| **AdamW** | Optimizer with weight decay |
| **DataLoader** | Batched clip loading from `.npy` frame stacks |
| **MobileNet baseline** | Optional lighter 2D-per-frame model for quick experiments |

Each training clip is **13 frames at 13 FPS** (~1 second), shaped **(C, T, H, W)** for the 3D model.

### Inference and post-processing

| Topic | What we used it for |
|-------|---------------------|
| **Sliding-window inference** | Scan video or live buffer with overlapping time windows |
| **Softmax** | Turn model logits into per-class probabilities |
| **Confidence thresholds** | Per-ability score cutoffs (stricter offline, looser live) |
| **Min-margin filter** | Winning class must beat the next-best ability by a margin |
| **Peak-only detection** | Emit only at local score peaks (offline; disabled live) |
| **Temporal NMS** | Suppress duplicate events within `nms_window_sec` |
| **Localize-then-classify** | Find champion first, crop, then run the neural net |
| **Localization cache** | Reuse last champion box briefly during live capture |

### Evaluation

| Topic | What we used it for |
|-------|---------------------|
| **Precision / recall / F1** | Per-class metrics on the validation clip split (scikit-learn) |
| **Confusion matrix** | See which abilities get confused during training |
| **Macro-F1** | Checkpoint selection over ability classes (excluding `background`) |
| **Event-level matching** | Match predicted cast timestamps to labels within ±0.5s tolerance |
| **TP / FP / FN** | Count true/false positives and missed casts per ability |

We tune for **recall first** on tracked abilities: missing a real cast is worse than an extra timer.

### Live system and web app

| Topic | What we used it for |
|-------|---------------------|
| **mss** | Real-time screen capture into a rolling frame buffer |
| **FastAPI** | Local web server for the timer UI |
| **WebSockets** | Push timer state to the browser ~4×/sec |
| **Threading** | Background capture + inference loop alongside the HTTP server |
| **Data Dragon API** | Sync ability base cooldowns from Riot's patch data (`sync_timer_cds.py`) |

### Data pipeline

| Topic | What we used it for |
|-------|---------------------|
| **Timestamp annotation** | Human labels at cast time → JSON event files |
| **Positive / negative sampling** | Cast-centered clips + random background windows |
| **Train/val split** | Stratified holdout per ability class |
| **YAML config** | One file drives classes, clip shape, localize, train, infer, timers |
| **Train == serve** | Same preprocessing in dataset build, recognize, and live |

## Development Narrative

### 1. Full-Screen Monitoring

The first version treated the full gameplay frame as the model input. We trained
a clip classifier on labeled casts and scanned videos with a sliding window. This
worked surprisingly well in controlled tests: the model learned Ezreal's VFX and
the offline evaluator reached near-perfect results on the training-style footage.

That success exposed the limitation. A full-screen classifier can look good when
the champion is near the same part of the screen as the training examples, but in
real games Ezreal can be anywhere: lane edges, fog fights, river, base, or small
on screen during camera movement. The spell VFX can also be surrounded by allied
or enemy spells, minions, HUD elements, and the minimap. The model was being
asked to solve two problems at once:

- find the specific champion in a noisy frame
- classify the ability VFX around that champion

The second task is a good fit for the neural network. The first task is better
handled before classification.

### 2. Frame Fitting and HUD Masking

Before localization, the preprocessing path was improved so train-time and
serve-time inputs matched exactly. The config supports `letterbox` mode, which
keeps the whole frame instead of center-cropping away the sides. We also added
HUD masks to black out regions such as the player's own ability bar and minimap.

This matters because the HUD can leak labels. For example, your own cooldown
sweep, mana changes, or minimap effects are cues that do not exist for an enemy
champion. Masking those areas makes the model learn VFX instead of shortcuts.

### 3. Name and Healthbar Localization

The major change was switching from "classify the whole screen" to
"find Ezreal, crop around him, then classify." The localizer combines classical
computer vision with the trained model:

- threshold red and green healthbar colors in HSV space
- filter candidate bars by shape, size, and aspect ratio
- ignore UI regions such as scoreboard, shop rail, HUD, and minimap
- template-match the champion name above the healthbar
- crop a fixed region below the healthbar and feed only that crop to the model

Name matching became the identity signal. Bar color alone is not enough because
the system should work on enemies, allies, or teammates. The nameplate says
"Ezreal"; the red or green bar just helps locate the champion-sized object.

The current localizer avoids full-frame name-template search as the primary
method because it produced texture false positives. Instead, it finds
champion-shaped bars first, then checks the smaller band above each bar for the
name template. This is both faster and less noisy.

### 4. Localize-Then-Classify Dataset Rebuild

Localization is not just a live inference trick. The dataset builder also uses
the localizer when `localize.enabled: true`. For every positive and background
clip, it samples the full-resolution frames, finds the champion, crops around
him, and saves the cropped clip.

That keeps training and inference aligned:

```text
raw video -> localize champion -> crop around healthbar -> train classifier
live screen -> localize champion -> crop around healthbar -> run classifier
```

Background examples are still important. They teach "Ezreal is visible but not
casting," instead of only teaching generic empty frames.

### 5. Model Fine-Tuning

The classifier is a short video model. The current primary backbone is `x3d_s`,
a pretrained video architecture loaded from PyTorchVideo. The final layer is
replaced with a new head for the champion-specific classes:

```text
background, Q, W, E, R, Flash
```

Training uses:

- fixed-length clips, currently 13 frames at 13 FPS, roughly one second
- cross-entropy loss with inverse-frequency class weights
- head warm-up for the first few epochs, where the pretrained backbone is frozen
- AdamW optimization
- validation with per-class precision, recall, F1, and a confusion matrix
- checkpoint selection by ability macro-F1, ignoring `background`

Q is still trained even though it is not shown in the timer UI. This is
intentional. Ezreal Q is a frequent visual event, and keeping it as a trained
class gives the model a "distractor sink." If Q were removed entirely, those
clips would have to be absorbed by `background`, `W`, `E`, `R`, or `Flash`, which
can increase false positives on the cooldowns we actually care about.

### Training Design Notes

This project is small-data transfer learning, not training a video model from
scratch. The clips are champion-specific, hand-labeled, and much smaller than
the datasets used to pretrain modern video backbones. That shaped most of the
training choices.

#### Why X3D-S

The primary model is `x3d_s`, a lightweight 3D convolutional video network from
PyTorchVideo. "3D" means the convolutions see both space and time: width, height,
and frame sequence. That is useful for League spells because many abilities are
not defined by a single still image. They have motion signatures: wind-up,
projectile direction, blink displacement, impact flash, and short-lived particle
effects.

X3D-S is a good fit here because it is:

- pretrained on action/video data, so it already has temporal feature priors
- smaller than many video architectures, which helps on limited hardware
- strong enough to model one-second VFX clips
- easy to replace the final classification head for our class list

The repo also keeps a `mobilenet_baseline`. That model processes each frame with
a 2D CNN and averages features over time. It is useful as a sanity check because
it is simpler and faster, but it is less expressive for short ability animations.

#### Transfer Learning Instead of Training From Scratch

Training from scratch would require far more labeled gameplay. Instead, the
pipeline loads a pretrained video model and replaces only the last projection
layer:

```text
pretrained X3D-S features -> new linear head -> background/Q/W/E/R/Flash logits
```

The early and middle layers already know general visual concepts such as motion,
edges, blobs, flashes, and object movement. Fine-tuning adapts those features to
League-specific spell VFX.

#### Head Warm-Up

For the first few epochs (`freeze_backbone_epochs: 3`), the backbone is frozen
and only the new classifier head trains. This gives the randomly initialized
head time to learn a reasonable mapping before gradients start changing the
pretrained feature extractor.

Without this step, the new head begins with random outputs, and early gradients
can push the pretrained backbone in noisy directions. Head warm-up is a cautious
way to avoid damaging useful pretrained features at the start of training.

After warm-up, the full model is unfrozen and fine-tuned end-to-end.

#### Learning Rate

The current learning rate is:

```yaml
lr: 0.0005
```

That is a conservative fine-tuning learning rate. It is high enough for the new
head to learn quickly, but low enough to avoid destroying pretrained X3D
features once the backbone is unfrozen. For this kind of project, the learning
rate is usually chosen by watching:

- training loss: should decrease without exploding
- validation macro-F1: should improve, then plateau
- per-class recall: tracked abilities should not collapse
- confusion matrix: one class should not absorb all others

If the model underfits, a higher learning rate or more epochs can help. If the
validation metrics jump around or get worse after unfreezing, the learning rate
is probably too high.

#### Adam vs AdamW

The optimizer is `AdamW`:

```python
torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
```

Adam adapts the step size per parameter, which is useful when fine-tuning a deep
network with a small and imperfect dataset. AdamW is the decoupled-weight-decay
version of Adam. In plain Adam, L2 regularization interacts with the adaptive
moment estimates. In AdamW, weight decay is applied more directly as a parameter
shrinkage term.

In practice, AdamW is usually preferred for modern neural-network fine-tuning
because it gives the benefits of Adam while making weight decay behave more like
regularization. That is why it is used here instead of SGD or plain Adam.

#### Weight Decay

The current weight decay is:

```yaml
weight_decay: 0.0001
```

Weight decay discourages overly large weights and helps reduce overfitting. That
matters because the dataset is generated from a limited number of recorded games.
Without regularization, the model can memorize recording-specific artifacts such
as camera position, HUD layout, compression noise, or repeated cast situations.

The value `1e-4` is a common starting point for fine-tuning pretrained vision
models. It is strong enough to regularize, but not so strong that it prevents the
model from adapting to League VFX.

#### Batch Size

The current batch size is:

```yaml
batch_size: 8
```

Video models are memory-heavy because every sample contains multiple frames.
`batch_size: 8` is a practical compromise: large enough for stable-ish gradient
updates, small enough to fit on typical local or cloud GPU memory for X3D-S.

If memory is available, a larger batch can make gradients smoother. If memory is
tight, a smaller batch still works, but validation metrics may be noisier.

#### Epoch Count and Checkpoint Selection

The config trains for:

```yaml
epochs: 25
```

The code does not simply keep the final epoch. It evaluates after each epoch and
saves `models/<champion>/best.pt` when ability macro-F1 improves. This matters
because the best validation model may happen before the final epoch, especially
on a small dataset where overfitting can start late in training.

The checkpoint stores not only weights, but also preprocessing metadata:

- class order
- model backbone
- normalization mean/std
- clip size, frame count, and sample FPS
- frame mode and HUD mask
- localization settings

That makes inference reproducible: the recognizer rebuilds the same model and
preprocessing path used during training.

#### Class Weighting

Training uses weighted cross-entropy:

```python
weights = class_weights(train_ds.label_ids, cfg.num_classes)
criterion = nn.CrossEntropyLoss(weight=weights)
```

The weights are inverse-frequency weights. If one class has fewer examples, each
mistake on that class matters more to the loss. This is important because the
dataset is not naturally balanced: `background`, Q, W, E, R, and Flash may have
different numbers of clips.

Without class weighting, the model can get a good average loss by becoming very
good at common classes while ignoring rare but important classes like Flash.

#### Metrics Used During Training

The training script reports precision, recall, F1, support, and a confusion
matrix for every class. It then selects checkpoints using **ability macro-F1**,
excluding `background`.

This is intentional. `background` is necessary for training, but it is not the
product outcome. The important question is whether the model can distinguish
actual abilities, especially tracked cooldowns. Macro-F1 gives each ability a
similar voice even if one class has more validation clips than another.

#### Why Validation Metrics Are Not the Whole Story

Clip-level validation tells us whether the model classifies sampled windows
correctly. The cooldown app needs event-level behavior: did a cast produce one
timer at the right time?

That is why the project also has `src.infer.evaluate`, which matches predicted
events against hand-labeled timestamps and reports TP/FP/FN, precision, recall,
and F1 per ability. A model can have strong clip metrics but still need threshold
tuning to avoid duplicate events or missed casts in the sliding-window output.

### 6. Threshold and Inference Tuning

After training, model probabilities are turned into discrete cast events. The
offline recognizer scans a video with a sliding window and applies:

- per-class confidence thresholds
- an optional `min_margin` over the next-best ability score
- optional `peak_only` filtering, so only local score peaks emit events
- temporal non-max suppression (`nms_window_sec`) to avoid duplicate events
- `infer.track`, which restricts which classes are emitted

The offline settings can be stricter because scanning every 0.1 seconds gives
dense probability curves and local peaks are easy to find. Live inference is
sparser and noisier, so it uses a separate `infer.live` block with lower
thresholds, `peak_only: false`, and cached localization.

The live settings reflect the recall-first philosophy: lower thresholds catch
more true casts, while `min_margin` and NMS claw back some precision without
hiding real events.

### 7. Cooldown Timer UI

The web app wraps the live detector with a cooldown engine. It serves a local UI
with FastAPI and streams timer state over WebSocket. Adding Ezreal starts the
detector if both `configs/ezreal.yaml` and `models/ezreal/best.pt` exist.

The UI only shows abilities and summoners that are actually tracked. For Ezreal,
that means `W`, `E`, `R`, and `Flash`, not every summoner spell and not Q.

Cooldowns now account for:

- ability rank
- ability haste
- summoner spell haste
- current Data Dragon base cooldowns synced into the YAML config

The formula is the League haste formula:

```text
effective cooldown = base cooldown * 100 / (100 + haste)
```

Ability base cooldowns live in `timers.abilities`. Summoner base cooldowns live
in the Python cooldown table. The script `scripts/sync_timer_cds.py` refreshes
ability base cooldowns from Data Dragon after Riot patches.

## Configuration Guide

The project is config-driven. `configs/ezreal.yaml` defines the classes, clip
shape, localization settings, training hyperparameters, inference thresholds,
live tuning, and cooldown timers.

### `champion`

The champion slug. It determines output paths such as:

- `data/clips/ezreal`
- `models/ezreal/best.pt`
- `configs/ezreal.yaml`

### `classes`

The classifier label order. `background` must be first. The order is stored in
the checkpoint and must match training and inference.

For Ezreal:

```yaml
classes:
  - background
  - Q
  - W
  - E
  - R
  - Flash
```

`Q` stays here because it improves discrimination, even though it is not emitted
as a tracked timer.

### `clip`

Controls how raw video is converted into model clips.

- `num_frames`: number of frames passed to the model.
- `sample_fps`: temporal sampling rate.
- `crop_size`: square spatial input size.
- `frame_mode`: `letterbox` keeps the whole frame; `center_crop` is legacy.
- `hud_mask`: fractional rectangles blacked out before classification.
- `spatial_jitter`: train-time random zoom/reposition to improve robustness.

Changing these requires rebuilding clips and retraining, because clips on disk
are already preprocessed.

### `localize`

Controls champion detection before classification.

- `enabled`: turns localize-then-classify on.
- `box_size`: crop size around the champion in source pixels.
- `offset_y`: vertical gap between healthbar and crop.
- `red_hsv` / `green_hsv`: HSV ranges for enemy and ally healthbars.
- `detect_green`: whether to include green bars.
- `bar_min_w`, `bar_max_w`, `bar_min_h`, `bar_max_h`: healthbar shape filters.
- `bar_min_aspect`: filters for long, thin bars instead of icons or text.
- `ignore_regions`: UI zones to skip.
- `name_first`: prefer name-template matching.
- `assume_single_enemy`: fallback mode for simple videos.
- `default_champion`: name used by fallback logic.
- `name_templates`: template images for champion nameplates.
- `name_match_threshold`: minimum template score.

This section is where most of the image-processing work lives.

### `dataset`

Controls clip extraction.

- `positive_offset_sec`: shifts positive windows relative to annotation time.
- `negatives_per_video`: number of background clips sampled per source video.
- `negative_min_gap_sec`: minimum distance from a labeled cast for negatives.
- `val_fraction`: held-out validation split.
- `seed`: reproducible train/val split and negative sampling.

### `train`

Controls fine-tuning.

- `backbone`: `x3d_s` is the current video model; `mobilenet_baseline` exists as
  a fast sanity-check model.
- `epochs`: total training epochs.
- `batch_size`: clips per optimization step.
- `lr`: learning rate.
- `weight_decay`: AdamW regularization.
- `num_workers`: data loading workers.
- `freeze_backbone_epochs`: warm up the new classifier head before unfreezing
  the pretrained backbone.

### `infer`

Controls offline event detection over recorded videos.

- `stride_sec`: how often to classify a window.
- `track`: classes that should be emitted as events.
- `default_threshold`: fallback confidence threshold.
- `thresholds`: per-class thresholds.
- `nms_window_sec`: suppress duplicate detections close together in time.
- `peak_only`: emit only local score peaks.
- `min_margin`: require the winning class to beat the next-best ability.

### `infer.live`

Overrides selected inference settings for real-time screen capture.

Live uses lower thresholds because inference is less dense than offline scanning.
It also disables `peak_only`, because a sparse live loop can miss the exact local
peak. `localize_cache_sec` reuses the most recent champion crop briefly, reducing
localizer overhead and smoothing over short localization misses.

### `timers`

Controls the web app timer surface.

- `abilities`: base cooldowns by ability rank.
- `summoners`: summoners this model can detect and this UI should show.
- `class_to_key`: optional remapping if a model class name differs from the timer
  key.

For Ezreal, `infer.track` plus `timers` means the UI shows only:

```text
W, E, R, Flash
```

## What This Project Demonstrates

This project combines several engineering skills in one pipeline:

- supervised dataset creation from video annotations
- transfer learning and fine-tuning of a pretrained video model
- train/validation metrics, confusion matrices, and checkpoint selection
- classical image processing with HSV masks, contour filtering, and template
  matching
- careful train/serve preprocessing consistency
- real-time screen capture with a rolling frame buffer
- threshold tuning, NMS, and event-level evaluation
- a small local web app with WebSocket-driven timer state
- product-aware ML tradeoffs, especially recall-first tuning for cooldown safety

The most important lesson from today's work was that high model accuracy in a
controlled setup was not enough. The practical system needed localization,
preprocessing discipline, and inference tuning so the model solved the right
problem: classifying the cast around Ezreal, not searching the entire game frame
for him at the same time.
