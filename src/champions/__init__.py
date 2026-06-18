"""Champion-specific Python extensions (use sparingly).

Most per-champion behavior belongs in ``configs/{champion}.yaml``:
classes, localize templates, infer thresholds, and ``timers.*`` blocks.
Add a submodule here only when YAML + ``src/timers/`` shared detectors are not
enough — e.g. a bespoke fusion rule or parser that cannot be reused.

Example layout when needed::

    src/champions/ahri/custom_thing.py
"""
