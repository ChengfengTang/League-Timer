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
- Only the **high-cooldown abilities worth timing** are emitted (see `infer.track`).
  Short-CD / spammed abilities (e.g. Ezreal Q) are intentionally not reported.

## Why spammed abilities stay in the model but aren't tracked

Keep short-CD abilities (like Ezreal Q) as **trained classes** even though we don't
report them. They act as a "distractor sink": when the champion casts that ability,
the model routes the score into that class instead of misfiring on a tracked
ability (W/E/R). The `infer.track` list then drops them from the output.

Removing such an ability from training entirely is usually a mistake — its casts
would have to be absorbed by `background`/W/E/R and can leak into the abilities we
*do* care about, creating exactly the false positives we were trying to avoid.
