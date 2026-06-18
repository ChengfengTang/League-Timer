"""Cast events and per-ability timer rules (``timers.on_cast`` in champion yaml).

All detection sources (VFX model, r_bar, manual UI) normalize to :class:`CastEvent`
and flow through :func:`apply_cast_event` before mutating cooldown slots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set

CastSource = Literal["vfx", "r_bar", "manual"]


@dataclass(frozen=True)
class CastEvent:
    ability: str
    source: CastSource
    time: Optional[float] = None


@dataclass
class ReduceOthersRule:
    sec: float
    targets: List[str]


@dataclass
class AbilityCastRules:
    start_cooldown: bool = False
    skip_if_ticking: bool = False
    reduce_others: Optional[ReduceOthersRule] = None


def _parse_reduce_others(raw: object) -> Optional[ReduceOthersRule]:
    if not isinstance(raw, dict):
        return None
    sec = raw.get("sec")
    targets = raw.get("targets")
    if sec is None or not targets:
        return None
    return ReduceOthersRule(float(sec), [str(t) for t in targets])


def _parse_ability_rules(raw: object) -> AbilityCastRules:
    if not isinstance(raw, dict):
        return AbilityCastRules()
    return AbilityCastRules(
        start_cooldown=bool(raw.get("start_cooldown", False)),
        skip_if_ticking=bool(raw.get("skip_if_ticking", False)),
        reduce_others=_parse_reduce_others(raw.get("reduce_others")),
    )


def load_cast_rules(infer: dict, timers: dict) -> Dict[str, AbilityCastRules]:
    """Build merged rules from ``infer.track`` / ``infer.emit`` and ``timers.on_cast``."""
    track: Set[str] = {str(k) for k in (infer.get("track") or [])}
    emit: Set[str] = {str(k) for k in (infer.get("emit") or [])}
    r_bar = timers.get("r_bar") or {}
    if r_bar.get("enabled"):
        track.add(str(r_bar.get("ability", "R")))

    rules: Dict[str, AbilityCastRules] = {}
    for key in track:
        rules[key] = AbilityCastRules(start_cooldown=True)
    for key in emit:
        rules.setdefault(key, AbilityCastRules())

    raw_on_cast = timers.get("on_cast") or {}
    for key, spec in raw_on_cast.items():
        key = str(key)
        parsed = _parse_ability_rules(spec)
        base = rules.get(key, AbilityCastRules())
        start = parsed.start_cooldown
        if isinstance(spec, dict) and "start_cooldown" in spec:
            start = bool(spec["start_cooldown"])
        elif key in track:
            start = True
        rules[key] = AbilityCastRules(
            start_cooldown=start,
            skip_if_ticking=parsed.skip_if_ticking or base.skip_if_ticking,
            reduce_others=parsed.reduce_others or base.reduce_others,
        )
    return rules


def apply_cast_event(
    champ,
    event: CastEvent,
    rules: Dict[str, AbilityCastRules],
    now: float,
    *,
    total_override: Optional[float] = None,
) -> bool:
    """Apply ``timers.on_cast`` rules for one cast event. Returns True if any slot changed."""
    key = champ.class_to_key.get(event.ability, event.ability)
    rule = rules.get(key)
    if rule is None:
        return False

    event_time = now if event.time is None else float(event.time)
    if event.source in ("vfx", "r_bar"):
        event_time -= champ.detection_lag_sec

    changed = False

    if rule.reduce_others is not None:
        for target in rule.reduce_others.targets:
            slot = champ.slot(target)
            if slot is not None and slot.reduce_remaining(now, rule.reduce_others.sec):
                changed = True

    if not rule.start_cooldown:
        return changed

    slot = champ.slot(key)
    if slot is None:
        return changed
    if rule.skip_if_ticking and slot.remaining(now) > 0.05:
        return changed

    if total_override is not None:
        eff = float(total_override)
    else:
        eff = champ._effective_total(slot)
    slot.trigger(event_time, eff)
    return True
