# League Timer

Computer-vision cooldown tracker for League of Legends: localize a champion on screen, detect ability casts, and run countdown timers in a live web UI.

## Language

**Champion profile**:
The per-champion bundle — `configs/{slug}.yaml`, optional `models/{slug}/best.pt`, and localization templates — that fully describes how one champion is detected and timed.
_Avoid_: Champion config (too vague), champion pack

**Detection source**:
Where a cast signal comes from before timer rules run: VFX model, HUD plugin (e.g. r_bar), or manual click.
_Avoid_: Detector, pipeline stage

**Cast event**:
A normalized `{ability, source, time}` signal where `source` is `vfx`, `r_bar`, or `manual`. All detection sources — including manual UI clicks — feed one pipeline; `timers.on_cast` rules decide slot changes.
_Avoid_: Detection, inference output

**Tracked ability**:
An ability in `infer.track` whose casts start a visible cooldown timer. Implicit default rule: `start_cooldown: true` unless `timers.on_cast` overrides.
_Avoid_: Monitored ability, timed spell

**Distractor class**:
An ability kept in the trained model (`classes:`) but excluded from `infer.track` so its VFX is absorbed without showing a timer in the UI. May still emit **cast events** for timer rules (e.g. Ezreal Q reducing other CDs).
_Avoid_: Ignored ability, spam class

**Emit-only class**:
A model class listed in `infer.emit` (not `infer.track`) that produces cast events and runs `timers.on_cast` rules but does not get a timer slot in the UI.
_Avoid_: Hidden track, silent detect

**Infer emit list**:
Config key `infer.emit` — abilities the VFX model may fire cast events for without tracking them in the UI. Disjoint from `infer.track`; a class should not appear in both.
_Avoid_: Secondary track, soft track

**Timer rule**:
Champion-specific logic in `timers.on_cast` applied when a cast event arrives — e.g. `start_cooldown`, `skip_if_ticking`, `reduce_others`. Same schema regardless of detection source. `reduce_others` only affects slots already ticking (`remaining > 0`).
_Avoid_: Cooldown handler, CD logic

**Shared detector plugin**:
Reusable non-model detection implemented once under `src/timers/` and enabled per champion via YAML (e.g. r_bar for ult charge icons).
_Avoid_: Generic detector, timer module

**Champion extension**:
Python under `src/champions/{slug}/` used only when YAML + shared plugins cannot express the behavior. Not built preemptively — add when a champion needs state or logic beyond declarative `timers.on_cast` rules (e.g. Ahri R charge/takedown state, if ever).
_Avoid_: Champion hack, special case file
