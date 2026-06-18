# Project guidance

## Detection philosophy: recall > precision

For this cooldown timer, **missing a real cast is worse than a false alarm.**
A late/extra timer is a minor annoyance; a *missing* timer means the user thinks
an ability is up when it isn't. Always bias tuning toward catching every cast of
the abilities we track.

Practical implications:
- Prefer **lower thresholds** on tracked abilities; accept some false positives.
- When trading off in `configs/<champion>.yaml` -> `infer`, optimize **recall first**,
  then claw back precision with `min_margin` / `peak_only` / `nms_window_sec`.
- Only the **high-cooldown abilities worth timing** get UI timers (see `infer.track`).
  Short-CD / spammed abilities (e.g. Ezreal Q) are intentionally not shown as timers.
- **`infer.emit`** is separate: emit-only classes fire cast events for **timer rules**
  (e.g. Ezreal Q shaving W/E/R) without a UI slot. Offline/CLI tools ignore emit today
  (detection-only there); the **web app** applies rules via `timers.on_cast`.

## Why spammed abilities stay in the model but aren't tracked

Keep short-CD abilities (like Ezreal Q) as **trained classes** even though we don't
show a timer for them. They act as a "distractor sink": when the champion casts that ability,
the model routes the score into that class instead of misfiring on a tracked
ability (W/E/R). Drop them from **`infer.track`** (UI timers); add to **`infer.emit`**
when they need side-effect rules (see Ezreal pattern in `docs/champion-tracking.md`).

Removing such an ability from training entirely is usually a mistake — its casts
would have to be absorbed by `background`/W/E/R and can leak into the abilities we
*do* care about, creating exactly the false positives we were trying to avoid.

---

## Champion onboarding procedure (for agents)

End-to-end playbook for adding or refreshing a champion. The **human** records
footage, annotates casts, and decides which abilities to auto-track; the **agent**
scaffolds config, runs build/train/validate loops, and updates docs.

Reference docs to keep in sync:
- [`docs/champion-tracking.md`](docs/champion-tracking.md) — per-champion patterns (track / emit / r_bar / on_cast)
- [`CONTEXT.md`](CONTEXT.md) — domain glossary (cast event, timer rule, emit-only class)
- [`README.md`](README.md) — command details and flags
- [`docs/technical-writeup.md`](docs/technical-writeup.md) — architecture background

**Web app vs offline/CLI:** Detection (VFX model, r_bar merge in recognize) is shared.
**Cooldown math and `timers.on_cast` rules apply only in the web app** (`src/app/cast_rules.py`).
Offline `recognize` / `evaluate` and `infer.live` CLI are spell-detection only — no timer simulation.

Slug convention: lowercase config name (`ezreal`, `ahri`) → paths
`configs/{slug}.yaml`, `data/{slug}/`, `models/{slug}/best.pt`.

### What the human must provide

Before the agent runs pipeline steps, get explicit answers (or infer from a similar champion):

1. **Champion slug** (e.g. `ahri`).
2. **Which abilities to auto-track** for cooldown timers — see decision guide below.
3. **Recorded video(s)** in `data/{slug}/raw_videos/` (human places files; never commit).
4. **Timestamp annotations** — human runs the annotator (or confirms annotations exist).
5. **Optional:** “R uses r_bar” vs “R uses VFX model”, drop Q from track (Ezreal-style
   → use `infer.emit` + `on_cast`), skill order for live rank scaling. **Flash stays on by default** unless human opts out.

### Champion patterns (pick closest template)

See [`docs/champion-tracking.md`](docs/champion-tracking.md). Common shapes:

| Pattern | `infer.track` | `infer.emit` | Other |
|---------|---------------|--------------|-------|
| **Standard kit** | Q/W/E/R/Flash | `[]` | — |
| **Distractor Q** (Ezreal) | W/E/R/Flash | `[Q]` | `timers.on_cast.Q.reduce_others` |
| **r_bar ult** (Ahri) | Q/W/E/Flash (no R) | `[]` | `timers.r_bar` + `on_cast.R.skip_if_ticking` |

Add **`src/champions/{slug}/`** Python only when yaml + `src/timers/` plugins cannot express the behavior (last resort).

### Default yaml scaffold (new champions)

**Start every new champion with full kit + Flash enabled.** Only remove entries when the human explicitly asks (or a known pattern like Ezreal Q distractor / Ahri r_bar R).

```yaml
classes:
  - background
  - Q
  - W
  - E
  - R
  - Flash

infer:
  track:
    - Q
    - W
    - E
    - R
    - Flash
  emit: []                    # emit-only classes (timer rules, no UI slot)
  thresholds:
    Q: 0.98
    W: 0.98
    E: 0.98
    R: 0.98
    Flash: 0.98
  live:
    thresholds:
      Q: 0.80
      W: 0.75
      E: 0.75
      R: 0.90
      Flash: 0.85

timers:
  summoners:
    - Flash
  # on_cast: {}               # optional per-ability rules (web app only); see patterns below
```

**Common exceptions** (apply only when human specifies or evidence demands it):

| Change | When |
|--------|------|
| Drop **Q** from `infer.track` | Short spam CD (Ezreal Q blocks other classes via `min_margin`); **keep Q in `classes:`** |
| Add **Q** to `infer.emit` + `on_cast` | Ezreal-style passive: Q detected → shave CDs off other abilities (no Q timer) |
| Drop **R** from `infer.track` | R timed via `timers.r_bar` instead of VFX (Ahri Spirit Rush); **keep R in `classes:`** |
| Add **`on_cast.R.skip_if_ticking`** | r_bar ult: charge icons flicker mid-CD — don't restart an active timer |
| Drop **Flash** from track | Rare — only if human says Flash detection isn't ready |

**`timers.on_cast` rule vocabulary** (web app, `src/app/cast_rules.py`):

- Implicit **`start_cooldown: true`** for every ability in `infer.track` (unless overridden).
- **`skip_if_ticking: true`** — ignore cast if that slot is already counting down.
- **`reduce_others: { sec: N, targets: [W, E, R] }`** — shave N seconds off ticking targets only.

After deciding, update **`docs/champion-tracking.md`** with the champion's final table.

Examples today: Ezreal → W/E/R/Flash tracked, Q emit-only (`reduce_others`). Ahri → Q/W/E/Flash tracked, R via r_bar + `skip_if_ticking`.

Everything else (yaml scaffold, templates from frames, build, train, evaluate, doc updates) is agent work unless the human asks to do it manually.

### Phase 0 — Agent: scaffold config

1. Copy the closest existing yaml → `configs/{slug}.yaml` (Ezreal: enemy tracking + Flash; Ahri: ally + r_bar pattern).
2. Set `champion: {slug}` and `classes:` (`background` first, then Q/W/E/R, add `Flash` if modeled).
3. Set `localize.default_champion` and `name_templates` path → `configs/templates/{slug}_name.png` (template created in phase 1).
4. Scaffold from the **default yaml** above (Q/W/E/R/Flash + `emit: []` + `timers.summoners: [Flash]`), then apply the closest **pattern** (standard / distractor Q / r_bar ult).
5. **Commit** `configs/{slug}.yaml` (yaml is tracked; `data/`, `models/`, `configs/templates/` are gitignored).

### Phase 1 — Human: record gameplay

Human saves recordings to `data/{slug}/raw_videos/`.

Recording guidelines (relay to human if missing footage):
- One champion per session; **1080p @ 30–60 fps**, consistent resolution across train and test.
- Cast **every ability many times** (aim ~50–150 clean casts per ability): varied targets, locations, skins.
- Include some teamfight / clutter footage; keep one video **held out** unmerged for final test.
- Same client **language** as nameplate templates (English name crop).

Videos stay local — do not commit (`data/` is gitignored).

### Phase 2 — Agent: localization assets

1. Pick a clear frame from raw video; crop champion **nameplate** → `configs/templates/{slug}_name.png`.
2. Run localizer preview and fix yaml until detection is stable:

```bash
python -m src.localize.preview --config configs/{slug}.yaml \
  --video data/{slug}/raw_videos/{stem}.mp4 \
  --out outputs/{slug}_loc_preview.mp4
```

3. Tune `localize:` (`box_size`, `ignore_regions`, `name_match_threshold`, green/red HSV) until red bar + green crop follow the target champion on most frames.
4. If using **r_bar** for an ability: capture a **single charge icon** (one swirl, not the full strip) from an R cast frame → `configs/templates/{slug}_r_icon.png`. Set `timers.r_bar.icon_template` and tune `match_threshold`, `height`, `icon_strip_fraction`.

Changing `localize:` or `clip.hud_mask` later requires **rebuild + retrain**.

### Phase 3 — Human: annotate videos

Human labels cast timestamps (agent can remind human of commands):

```bash
python -m src.annotate.annotate --config configs/{slug}.yaml \
  --video data/{slug}/raw_videos/{stem}.mp4
```

Controls: SPACE play/pause, `,`/`.` frame step, `[`/`]` ±1s, letter keys for abilities (from config classes), `u` undo, `x` delete-near, `s` save, ESC quit.

Output: `data/{slug}/annotations/{stem}.json`. Re-running resumes existing labels.

**Agent:** after human finishes, skim annotation counts per ability; flag sparse classes (especially R/Flash) and suggest more recording if any tracked ability has &lt; ~20 labels.

### Phase 4 — Agent: cooldowns in yaml

```bash
python scripts/sync_timer_cds.py {slug}
```

- Updates `timers.abilities` from Data Dragon for keys in **`infer.track`** (and r_bar ability when enabled; skips Q and summoners).
- **Does not** auto-add abilities tracked only via **r_bar** when r_bar is disabled. Add those CDs manually under `timers.abilities` or extend the sync script.
- Pin patch if needed: `--version 16.12.1`.
- Set `timers.summoners`, `timers.skill_order`, and `timers.on_cast` per human request / pattern.

### Phase 5 — Agent: build clip dataset

```bash
python -m src.dataset.build --config configs/{slug}.yaml
```

Writes `data/{slug}/clips/frames/` + `manifest.json`. Requires annotations for all videos under `data/{slug}/annotations/`.

If `localize:` or `clip.*` changed since last build, delete old clips first or rebuild over them.

**Colab upload prep** (clips only, not full repo):

```bash
cd data/{slug} && zip -r {slug}_clips.zip clips/
```

Human uploads `{slug}_clips.zip` in Colab when prompted (`scripts/train.ipynb`).

### Phase 6 — Agent: train

**Local (Apple Silicon — slow):**

```bash
python -m src.train.train --config configs/{slug}.yaml --device mps
```

**Colab (recommended):**

1. Open `scripts/train.ipynb`; set `CHAMPION = "{slug}"` and `REPO_URL`.
2. Run all cells; upload `{slug}_clips.zip` when prompted.
3. Download `models/{slug}/best.pt` into local `models/{slug}/`.

Checkpoint path: `models/{slug}/best.pt` (gitignored). Review validation per-class precision/recall printed during training.

### Phase 7 — Agent: offline validate

Run on **train** and **held-out** videos:

```bash
# Events JSON (+ r_bar merge when enabled)
python -m src.infer.recognize --config configs/{slug}.yaml \
  --checkpoint models/{slug}/best.pt \
  --video data/{slug}/raw_videos/{stem}.mp4 \
  --out-dir outputs/{slug}

# Visual: bar, crop, level ROI, magenta r_bar box
python -m src.infer.preview_combined --config configs/{slug}.yaml \
  --checkpoint models/{slug}/best.pt \
  --video data/{slug}/raw_videos/{stem}.mp4 \
  --out-dir outputs/{slug}

# Scores vs human labels (model + r_bar abilities in infer.track; emit-only classes optional)
python -m src.infer.evaluate --config configs/{slug}.yaml \
  --pred outputs/{slug}/{stem}.events.json \
  --truth data/{slug}/annotations/{stem}.json
```

Bias **recall first** when tuning `infer.thresholds`, `min_margin`, `peak_only`, `nms_window_sec`. Separate **`infer.live`** block for web app / screen capture.

For **r_bar** ultimates: confirm first cast starts timer in the **web app**, charge spend (3→2→1) and takedown refresh (1→2) do **not** restart CD; tune `rearm_after_absent_sec` if double-fires or missed second R. Validate r_bar in `preview_combined` (magenta ROI).

### Phase 8 — Finetune loop (agent + human)

If evaluate shows misses or excess false positives:

1. **Human:** record more footage targeting weak abilities; annotate new casts.
2. **Agent:** `src.dataset.build` again → retrain → re-run recognize / evaluate / preview_combined.
3. Adjust yaml thresholds (recall first, then precision).
4. Update `docs/champion-tracking.md` if track / emit / on_cast / r_bar changed.

Repeat until tracked abilities meet acceptable recall on held-out video.

### Phase 9 — Live / web app smoke test

```bash
python -m src.app.server   # http://127.0.0.1:8000
```

Add champion in UI (needs `configs/{slug}.yaml` + `models/{slug}/best.pt`). macOS: grant Screen Recording to terminal/IDE.

Verify **timer rules** in the web app (not offline): standard start-CD, Ezreal Q shave, Ahri R skip-if-ticking when r_bar enabled.

Note: **`infer.live` CLI** prints detections (including `infer.emit` classes and r_bar) but does **not** run cooldown math — timer rules are web-app only.

### Git / privacy checklist

| Path | Commit? |
|------|---------|
| `configs/{slug}.yaml` | Yes |
| `docs/champion-tracking.md`, `AGENTS.md` | Yes |
| `data/`, `models/`, `outputs/`, `configs/templates/`, `*.zip` | **No** (gitignored) |

Do not commit unless the user explicitly asks.

### Agent quick checklist (copy for each new champion)

```
[ ] Human: slug + track/emit pattern + videos in data/{slug}/raw_videos/
[ ] Agent: configs/{slug}.yaml from nearest pattern (standard / Ezreal / Ahri)
[ ] Agent: name template (+ r_icon if r_bar)
[ ] Agent: localize.preview passes on sample video
[ ] Human: annotate all training videos
[ ] Agent: sync_timer_cds (+ manual CDs for r_bar-only abilities + on_cast if needed)
[ ] Agent: dataset.build
[ ] Human: upload {slug}_clips.zip to Colab (if cloud train)
[ ] Agent: train → models/{slug}/best.pt
[ ] Agent: recognize + preview_combined + evaluate on train + holdout
[ ] Agent: tune infer thresholds (recall first)
[ ] Agent: update docs/champion-tracking.md
[ ] Agent: web app smoke test
```

## Agent skills

### Issue tracker

GitHub Issues on `ChengfengTang/League-Timer` via the `gh` CLI; external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles with default label strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `CONTEXT.md` and `docs/adr/` at the repo root (created lazily by domain-modeling skills). See `docs/agents/domain.md`.

