"""Map champion level to ability ranks from a configured skill-order path."""
from __future__ import annotations

from typing import Dict, List, Optional


def parse_skill_order(raw: Optional[Dict]) -> Dict[int, Dict[str, int]]:
    """Parse ``timers.skill_order`` from config.

    Format::

        skill_order:
          8: {E: 2}
          10: {E: 3}
          11: {R: 2}

    Keys are champion levels; values set ability ranks *at or above* that level.
    Levels below the first entry start at rank 1 for every listed ability.
    """
    if not raw:
        return {}
    out: Dict[int, Dict[str, int]] = {}
    for lvl, ranks in raw.items():
        level = int(lvl)
        out[level] = {str(k).upper(): int(v) for k, v in (ranks or {}).items()}
    return out


def ability_keys_from_skill_order(skill_order: Dict[int, Dict[str, int]]) -> List[str]:
    keys: List[str] = []
    for ranks in skill_order.values():
        for k in ranks:
            if k not in keys:
                keys.append(k)
    return keys


def ranks_at_level(level: int, skill_order: Dict[int, Dict[str, int]]) -> Dict[str, int]:
    """Return ability ranks implied by champion level and the skill-order table."""
    level = min(18, max(1, int(level)))
    keys = ability_keys_from_skill_order(skill_order)
    ranks = {k: 1 for k in keys}
    for lvl in sorted(skill_order):
        if lvl > level:
            break
        for key, rank in skill_order[lvl].items():
            ranks[key] = max(ranks.get(key, 1), rank)
    return ranks
