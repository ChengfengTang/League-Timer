"""Cooldown tracking engine for the League Timer web app.

Tracks champions with ability + summoner timers. Cooldown duration uses the
League formula from the original app::

    effective = base * 100 / (100 + haste)

Abilities pick their base CD from a per-rank table; summoners use a flat base
CD with summoner spell haste.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Dict, List, Optional, Union

# Base cooldowns for summoners we might track (seconds).
SUMMONER_BASE: Dict[str, float] = {
    "Flash": 300,
    "Ignite": 180,
    "Teleport": 360,
    "Ghost": 210,
    "Heal": 240,
    "Barrier": 180,
    "Exhaust": 210,
    "Cleanse": 210,
    "Smite": 90,
}


def effective_cooldown(base: float, haste: float) -> float:
    """League ability / summoner haste formula."""
    if haste <= 0:
        return float(base)
    return float(base) * 100.0 / (100.0 + float(haste))


def _parse_cooldowns(value: Union[float, int, List]) -> List[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def base_cd_at_rank(cooldowns: List[float], rank: int) -> float:
    """Rank is 1-based. A single-entry table applies to every rank."""
    if not cooldowns:
        return 0.0
    if len(cooldowns) == 1:
        return cooldowns[0]
    idx = min(max(rank, 1), len(cooldowns)) - 1
    return cooldowns[idx]


class _Slot:
    __slots__ = ("key", "label", "kind", "base_cooldowns", "rank", "total", "ends_at")

    def __init__(self, key: str, label: str, kind: str,
                 base_cooldowns: List[float], rank: int = 1) -> None:
        self.key = key
        self.label = label
        self.kind = kind  # "ability" | "summoner"
        self.base_cooldowns = base_cooldowns
        self.rank = rank
        self.total = 0.0
        self.ends_at: Optional[float] = None

    def base_cd(self) -> float:
        return base_cd_at_rank(self.base_cooldowns, self.rank)

    def trigger(self, now: float, total: float) -> None:
        self.total = float(total)
        self.ends_at = now + self.total

    def reset(self) -> None:
        self.ends_at = None
        self.total = 0.0

    def remaining(self, now: float) -> float:
        if self.ends_at is None:
            return 0.0
        return max(0.0, self.ends_at - now)

    def view(self, now: float, effective_cd: Optional[float] = None) -> Dict:
        rem = self.remaining(now)
        eff = effective_cd if effective_cd is not None else self.base_cd()
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "rank": self.rank,
            "base_cd": round(self.base_cd(), 1),
            "effective_cd": round(eff, 1),
            "total": round(self.total, 1),
            "remaining": round(rem, 1),
            "status": "ticking" if rem > 0.05 else "ready",
        }


class _Champion:
    def __init__(
        self,
        name: str,
        ability_cooldowns: Dict[str, Union[float, List[float]]],
        summoner_keys: List[str],
        auto: bool,
        class_to_key: Optional[Dict[str, str]] = None,
    ) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.name = name
        self.auto = auto
        self.level = 1
        self.ability_haste = 0
        self.summoner_haste = 0
        self.detector_status: Dict = {}
        self.class_to_key = dict(class_to_key or {})

        self.abilities: Dict[str, _Slot] = {
            key: _Slot(key, key, "ability", _parse_cooldowns(cd))
            for key, cd in ability_cooldowns.items()
        }
        self.summoners: Dict[str, _Slot] = {}
        for key in summoner_keys:
            base = SUMMONER_BASE.get(key, 300.0)
            self.summoners[key] = _Slot(key, key, "summoner", [base])

    def slot(self, key: str) -> Optional[_Slot]:
        return self.abilities.get(key) or self.summoners.get(key)

    def _effective_total(self, slot: _Slot) -> float:
        haste = self.summoner_haste if slot.kind == "summoner" else self.ability_haste
        return effective_cooldown(slot.base_cd(), haste)

    def view(self, now: float) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "auto": self.auto,
            "level": self.level,
            "ability_haste": self.ability_haste,
            "summoner_haste": self.summoner_haste,
            "detector_status": self.detector_status,
            "abilities": [
                self.abilities[k].view(
                    now, effective_cooldown(self.abilities[k].base_cd(), self.ability_haste))
                for k in self.abilities
            ],
            "summoners": [
                self.summoners[k].view(
                    now, effective_cooldown(self.summoners[k].base_cd(), self.summoner_haste))
                for k in self.summoners
            ],
        }


class CooldownEngine:
    """Thread-safe collection of champions and their cooldown timers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._champions: Dict[str, _Champion] = {}

    def add_champion(
        self,
        name: str,
        ability_cooldowns: Optional[Dict[str, Union[float, List[float]]]] = None,
        summoner_keys: Optional[List[str]] = None,
        auto: bool = False,
        class_to_key: Optional[Dict[str, str]] = None,
    ) -> Dict:
        with self._lock:
            champ = _Champion(
                name, ability_cooldowns or {}, summoner_keys or [], auto, class_to_key)
            self._champions[champ.id] = champ
            return champ.view(time.monotonic())

    def remove_champion(self, champion_id: str) -> bool:
        with self._lock:
            return self._champions.pop(champion_id, None) is not None

    def trigger(self, champion_id: str, key: str,
                total: Optional[float] = None) -> bool:
        now = time.monotonic()
        with self._lock:
            champ = self._champions.get(champion_id)
            if champ is None:
                return False
            slot = champ.slot(key)
            if slot is None:
                return False
            eff = total if total is not None else champ._effective_total(slot)
            slot.trigger(now, eff)
            return True

    def reset(self, champion_id: str, key: str) -> bool:
        with self._lock:
            champ = self._champions.get(champion_id)
            if champ is None:
                return False
            slot = champ.slot(key)
            if slot is None:
                return False
            slot.reset()
            return True

    def on_detection(self, champion_id: str, ability_class: str) -> bool:
        with self._lock:
            champ = self._champions.get(champion_id)
            if champ is None:
                return False
            key = champ.class_to_key.get(ability_class, ability_class)
            slot = champ.slot(key)
            if slot is None:
                return False
            slot.trigger(time.monotonic(), champ._effective_total(slot))
            return True

    def set_level(self, champion_id: str, level: int) -> bool:
        with self._lock:
            champ = self._champions.get(champion_id)
            if champ is None:
                return False
            champ.level = min(18, max(1, int(level)))
            return True

    def set_ability_haste(self, champion_id: str, haste: int) -> bool:
        with self._lock:
            champ = self._champions.get(champion_id)
            if champ is None:
                return False
            champ.ability_haste = max(0, int(haste))
            return True

    def set_summoner_haste(self, champion_id: str, haste: int) -> bool:
        with self._lock:
            champ = self._champions.get(champion_id)
            if champ is None:
                return False
            champ.summoner_haste = max(0, int(haste))
            return True

    def set_ability_rank(self, champion_id: str, key: str, rank: int) -> bool:
        with self._lock:
            champ = self._champions.get(champion_id)
            if champ is None:
                return False
            slot = champ.abilities.get(key)
            if slot is None:
                return False
            slot.rank = min(5, max(1, int(rank)))
            return True

    def set_detector_status(self, champion_id: str, status: Dict) -> None:
        with self._lock:
            champ = self._champions.get(champion_id)
            if champ is not None:
                champ.detector_status = status

    def set_auto(self, champion_id: str, value: bool) -> None:
        with self._lock:
            champ = self._champions.get(champion_id)
            if champ is not None:
                champ.auto = value

    def snapshot(self) -> List[Dict]:
        now = time.monotonic()
        with self._lock:
            return [c.view(now) for c in self._champions.values()]
