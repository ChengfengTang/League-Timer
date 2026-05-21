import type { VoiceAction } from "../types";
import type { SpellsStore } from "./spells";

const SUMMONER =
  "flash|ignite|teleport|tp|ghost|heal|barrier|exhaust|cleanse|smite";

function abilityAction(
  spells: SpellsStore,
  champion: string,
  token: string,
  confirmHit: boolean,
): VoiceAction[] | null {
  const resolved = spells.resolveChampionName(champion);
  if (!resolved) return null;
  const ab =
    spells.resolveAbility(resolved, token) ??
    spells.resolveSummoner(token);
  if (!ab) return null;
  return [
    {
      action: confirmHit ? "ability_hit" : "ability_used",
      champion: resolved,
      ability: ab,
      confirm_hit: confirmHit,
    },
  ];
}

export function parseVoiceRegex(
  transcript: string,
  spells: SpellsStore,
): VoiceAction[] | null {
  const text = transcript.trim();
  if (!text) return null;

  let m = text.match(/^add champion (.+)$/i);
  if (m) {
    const name = spells.resolveChampionName(m[1].trim());
    if (name) return [{ action: "add_champion", champion: name }];
  }

  m = text.match(/^remove (.+)$/i);
  if (m) {
    const name = spells.resolveChampionName(m[1].trim());
    if (name) return [{ action: "remove_champion", champion: name }];
  }

  m = text.match(new RegExp(`^(.+?) used ([QWER]|${SUMMONER})$`, "i"));
  if (m) return abilityAction(spells, m[1], m[2], false);

  m = text.match(new RegExp(`^(.+?) no ([QWER]|${SUMMONER})$`, "i"));
  if (m) return abilityAction(spells, m[1], m[2], false);

  m = text.match(/^(.+?) ([QWER]) landed$/i);
  if (m) return abilityAction(spells, m[1], m[2], true);

  m = text.match(/^(.+?) ([QWER]) down$/i);
  if (m) return abilityAction(spells, m[1], m[2], false);

  m = text.match(/^(.+?) level (\d+)$/i);
  if (m) {
    const name = spells.resolveChampionName(m[1].trim());
    if (name)
      return [{ action: "set_level", champion: name, level: parseInt(m[2], 10) }];
  }

  m = text.match(/^(.+?) ability haste (\d+)$/i);
  if (m) {
    const name = spells.resolveChampionName(m[1].trim());
    if (name)
      return [
        {
          action: "set_ability_haste",
          champion: name,
          ability_haste: parseInt(m[2], 10),
        },
      ];
  }

  m = text.match(/^(\d+) ability haste(?: on)? (.+)$/i);
  if (m) {
    const name = spells.resolveChampionName(m[2].trim());
    if (name)
      return [
        {
          action: "set_ability_haste",
          champion: name,
          ability_haste: parseInt(m[1], 10),
        },
      ];
  }

  m = text.match(/^(.+?) ([QWER]) rank (\d+)$/i);
  if (m) {
    const name = spells.resolveChampionName(m[1].trim());
    if (name)
      return [
        {
          action: "set_ability_rank",
          champion: name,
          ability: m[2].toUpperCase(),
          rank: parseInt(m[3], 10),
        },
      ];
  }

  m = text.match(/^(.+?) ([QWER]) back up$/i);
  if (m) {
    const name = spells.resolveChampionName(m[1].trim());
    if (name)
      return [
        {
          action: "ability_ready_ack",
          champion: name,
          ability: m[2].toUpperCase(),
        },
      ];
  }

  m = text.match(/^(.+?) (\w+(?:\s+\w+)?) down$/i);
  if (m) {
    const name = spells.resolveChampionName(m[1].trim());
    if (name) {
      const ab = spells.resolveAbility(name, m[2].trim());
      if (ab) return [{ action: "ability_used", champion: name, ability: ab }];
    }
  }

  return null;
}
