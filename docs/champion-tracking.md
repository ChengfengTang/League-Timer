# Champion tracking reference

Living doc: **what each champion auto-detects**, how timers start, and which config knobs control it.
Update this file whenever you add a champion or change `infer.track` / `timers.r_bar`.

Full onboarding steps (record → annotate → build → train → validate): see [`AGENTS.md`](../AGENTS.md#champion-onboarding-procedure-for-agents).

## How the system chooses a detector

Everything is **config-driven** from `configs/{champion}.yaml`. There is no hard-coded “if Ahri” logic in Python.

| Path | Config | Code |
|------|--------|------|
| **VFX model** (Q/W/E/R/Flash clips) | `infer.track` — only listed classes are emitted | `src/infer/recognize.py`, `src/app/detector.py`, `src/infer/live.py` |
| **R charge icons under bar** | `timers.r_bar.enabled: true` (+ template/thresholds) | `src/timers/r_bar.py` via `load_r_bar(cfg)` |

**Ezreal does not use r_bar.** His yaml has no `timers.r_bar` block, so `load_r_bar()` returns `None` and the icon detector never runs.

**Ahri uses r_bar for R.** R is intentionally **not** in `infer.track`; Spirit Rush is detected when charge swirls appear under the mana bar (rising edge: nothing → icons).

### `load_r_bar` gate

```python
# src/timers/__init__.py
def load_r_bar(cfg, ...):
    raw = cfg.section("timers").get("r_bar")
    return RBarDetector.from_config(raw, ...)  # None if missing or enabled: false
```

Call sites always check `if r_bar is not None` before scanning or updating.

### Where r_bar runs today

| Entry point | Uses r_bar? |
|-------------|-------------|
| `python -m src.infer.recognize` | Yes — merges r_bar events after model scan |
| `python -m src.infer.preview_combined` | Yes — magenta ROI + CAST R overlay |
| Web app (`src/app/detector.py` + `src/app/server.py`) | Yes — live loop + R on timer card when enabled |
| `python -m src.infer.live` (CLI) | **No** — model only; Ahri R won't auto-print here yet |

The web app also adds R to the timer UI when `timers.r_bar.enabled` is true, even if R is absent from `infer.track` (`server.timer_spec_from_config`).

### Classes in the model vs what we report

All abilities stay in `classes:` for training (short-CD spam like Ezreal Q acts as a **distractor sink** — see `AGENTS.md`). Only `infer.track` (plus optional r_bar) controls **UI timers**. **`infer.emit`** fires cast events for timer rules without a UI slot (Ezreal Q → `reduce_others` on W/E/R).

Cast events from VFX, r_bar, and manual clicks all flow through **`timers.on_cast`** rules in the web app (`src/app/cast_rules.py`).

---

## Ezreal

**Config:** `configs/ezreal.yaml`

| Ability | In model | Auto-tracked | Notes |
|---------|----------|--------------|-------|
| Q | Yes | **Emit-only** (`infer.emit`) | No UI timer; `on_cast` shaves 1.5s off ticking W/E/R (Rising Spell Force). |
| W | Yes | Yes | Primary tracked spell. |
| E | Yes | Yes | Arcane Shift — visually similar to Flash; still noisy vs Flash in some footage (tune thresholds). |
| R | Yes | Yes | VFX model. |
| Flash | Yes | Yes | Summoner in `timers.summoners`. |

**R bar:** off (no `timers.r_bar` block).

**Timer UI:** W, E, R, Flash.

---

## Ahri

**Config:** `configs/ahri.yaml`

| Ability | In model | Auto-tracked | Notes |
|---------|----------|--------------|-------|
| Q | Yes | Yes | Short CD — expect some FPs; recall-first tuning. |
| W | Yes | Yes | |
| E | Yes | Yes | |
| R | Yes (trained) | **r_bar only** | VFX model not used for timers; 3 dashes + refresh — CD starts on **0 icons → icons appear**, not on charge count changes. |
| Flash | Yes | Yes | Summoner in `timers.summoners`. |

**R bar:** `timers.r_bar.enabled: true` — template `configs/templates/ahri_r_icon.png`, rising-edge + `rearm_after_absent_sec`. `on_cast.R.skip_if_ticking: true` ignores charge flicker mid-CD.

**Timer UI:** Q, W, E, R (R from r_bar), Flash.

---

## Adding or changing a champion

1. Copy a yaml; pick the closest **pattern** (see table below).
2. Start from **Q/W/E/R + Flash** in `infer.track`, `infer.emit: []`, and `timers.summoners: [Flash]` (see AGENTS.md default scaffold).
3. Apply pattern-specific changes (distractor Q, r_bar ult, `on_cast` rules).
4. If ult needs **HUD icons instead of VFX**, add `timers.r_bar` with `enabled: true`, **remove R from `infer.track`**, add `on_cast.R.skip_if_ticking`.
5. Set `timers.abilities` CDs; run `python scripts/sync_timer_cds.py {champion}` after patches.
6. Update **this file** with a row for the new champion.

### Pattern reference

| Pattern | Example | `infer.track` | `infer.emit` | Notes |
|---------|---------|---------------|--------------|-------|
| Standard kit | most champions | Q/W/E/R/Flash | `[]` | implicit start CD |
| Distractor Q + passive | Ezreal | W/E/R/Flash | `[Q]` | `on_cast.Q.reduce_others` |
| r_bar ult | Ahri | Q/W/E/Flash | `[]` | `timers.r_bar` + `on_cast.R.skip_if_ticking` |

Champion-specific Python (`src/champions/{slug}/`) is **last resort** when yaml +
`src/timers/` plugins are not enough. See [`CONTEXT.md`](../CONTEXT.md).

### r_bar snippet (when needed)

```yaml
timers:
  on_cast:
    R:
      skip_if_ticking: true
  r_bar:
    enabled: true
    icon_template: configs/templates/{champion}_r_icon.png
    match_threshold: 0.62
    rearm_after_absent_sec: 1.5
  abilities:
    R: [140, 120, 100]
```

### emit-only + on_cast snippet (Ezreal-style passive)

```yaml
infer:
  track: [W, E, R, Flash]
  emit: [Q]

timers:
  on_cast:
    Q:
      reduce_others: { sec: 1.5, targets: [W, E, R] }
```

---

## Changelog

| Date | Change |
|------|--------|
| 2026-06-18 | Unified cast-event pipeline (web app): `infer.emit`, `timers.on_cast`, `src/app/cast_rules.py`. Docs synced across AGENTS/README/technical-writeup. |
| 2026-06-15 | Ahri: Q/W/E/Flash + r_bar R. Ezreal: W/E/R/Flash, Q emit-only. Default scaffold: QWER + Flash. |
