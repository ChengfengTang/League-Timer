#!/usr/bin/env python3
"""Sync timers.abilities in a champion config from live Data Dragon CDs.

The web app reads base cooldowns from configs/{ChampionName}.yaml (timers.abilities).
This script fetches the current patch from ddragon and updates only the abilities
listed in infer.track (excluding Q and summoners).

Usage::

    python scripts/sync_timer_cds.py ezreal
    python scripts/sync_timer_cds.py ezreal --version 16.12.1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Union

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
ABILITY_KEYS = ["Q", "W", "E", "R"]


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def latest_version() -> str:
    versions = fetch_json("https://ddragon.leagueoflegends.com/api/versions.json")
    return versions[0]


def parse_cooldowns(cooldown_burn: str) -> Union[int, List[float]]:
    if not cooldown_burn or cooldown_burn == "0":
        return 0
    parts = [float(x.strip()) for x in cooldown_burn.split("/")]
    if len(parts) == 1:
        v = parts[0]
        return int(v) if v == int(v) else v
    return parts


def ddragon_abilities(champion_id: str, version: str) -> Dict[str, Union[int, float, List[float]]]:
    url = (
        f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/"
        f"champion/{champion_id}.json"
    )
    data = fetch_json(url)["data"][champion_id]
    out: Dict[str, Union[int, float, List[float]]] = {}
    for i, key in enumerate(ABILITY_KEYS):
        spell = data["spells"][i]
        out[key] = parse_cooldowns(spell["cooldownBurn"])
    return out


def champion_id_for_slug(slug: str, version: str) -> str:
    """Map config slug to Data Dragon champion id (e.g. ezreal -> Ezreal)."""
    index = fetch_json(
        f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
    )
    slug_l = slug.lower()
    for brief in index["data"].values():
        if brief["id"].lower() == slug_l or brief["name"].lower().replace(" ", "") == slug_l:
            return brief["id"]
    raise SystemExit(f"Champion not found in Data Dragon: {slug}")


def tracked_ability_keys(cfg: dict) -> List[str]:
    infer = cfg.get("infer") or {}
    timers = cfg.get("timers") or {}
    track = {str(k) for k in (infer.get("track") or [])}
    summoners = {str(s) for s in (timers.get("summoners") or [])}
    keys = []
    for key in ("W", "E", "R", "Q"):
        if key in summoners or key == "Q":
            continue
        if track and key not in track:
            continue
        keys.append(key)
    return keys


def format_cd(value: Union[int, float, List]) -> str:
    if isinstance(value, list):
        inner = ", ".join(str(int(v) if v == int(v) else v) for v in value)
        return f"[{inner}]"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def replace_abilities_block(text: str, abilities: Dict[str, Union[int, float, List]]) -> str:
    """Replace lines under timers.abilities: without touching the rest of the file."""
    lines = text.splitlines(keepends=True)
    start = None
    end = None
    for i, line in enumerate(lines):
        if re.match(r"^  abilities:\s*$", line):
            start = i + 1
            continue
        if start is not None and re.match(r"^  \w", line):
            end = i
            break
    if start is None:
        raise SystemExit("Could not find timers.abilities: block in config")

    indent = "    "
    new_lines = [f"{indent}{key}: {format_cd(val)}\n" for key, val in abilities.items()]
    return "".join(lines[:start] + new_lines + lines[end:])


def sync(slug: str, version: str | None) -> None:
    config_path = CONFIGS / f"{slug}.yaml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    version = version or latest_version()
    champ_id = champion_id_for_slug(slug, version)
    ddragon = ddragon_abilities(champ_id, version)

    with open(config_path) as f:
        text = f.read()
    cfg = yaml.safe_load(text) or {}

    keys = tracked_ability_keys(cfg)
    if not keys:
        raise SystemExit(f"No tracked abilities to sync for {slug}")

    abilities = {k: ddragon[k] for k in keys}
    print(f"Data Dragon {version} — {champ_id}")
    for k, v in abilities.items():
        print(f"  {k}: {v}")

    patch_line = f"# (patch {version})"
    if re.search(r"^# \(patch [\d.]+\)\s*$", text, re.M):
        text = re.sub(r"^# \(patch [\d.]+\)\s*$", patch_line, text, count=1, flags=re.M)
    elif "sync_timer_cds.py" in text:
        text = re.sub(
            r"(sync_timer_cds\.py \w+\n)",
            rf"\1{patch_line}\n",
            text,
            count=1,
        )

    updated = replace_abilities_block(text, abilities)
    config_path.write_text(updated)
    print(f"Updated {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", help="Champion config slug (e.g. ezreal)")
    parser.add_argument("--version", help="Data Dragon version (default: latest)")
    args = parser.parse_args()
    sync(args.slug, args.version)


if __name__ == "__main__":
    main()
