import type { ChampionSpells, SpellsFile, SpellData } from "../types";
import { SUMMONER_ALIASES, SUMMONER_SPELLS } from "./summoners";

let store: SpellsStore | null = null;

export class SpellsStore {
  meta: Record<string, string> = {};
  champions: Record<string, ChampionSpells> = {};
  aliasToName = new Map<string, string>();
  abilityAliases = new Map<string, string>();

  static async load(): Promise<SpellsStore> {
    if (store) return store;
    const res = await fetch("/spells.json");
    if (!res.ok) throw new Error("Missing /spells.json — run npm run ingest");
    const file = (await res.json()) as SpellsFile;
    const s = new SpellsStore();
    s.meta = file.meta ?? {};
    s.champions = file.champions ?? {};

    for (const [name, champ] of Object.entries(s.champions)) {
      s.aliasToName.set(name.toLowerCase(), name);
      s.aliasToName.set(champ.key.toLowerCase(), name);
      for (const alias of champ.aliases ?? []) {
        s.aliasToName.set(alias.toLowerCase(), name);
      }
      for (const ab of ["Q", "W", "E", "R"] as const) {
        const spell = champ[ab];
        if (!spell) continue;
        s.abilityAliases.set(
          `${name.toLowerCase()}|${spell.name.toLowerCase()}`,
          ab,
        );
        const lastWord = spell.name.split(/\s+/).pop()?.toLowerCase();
        if (lastWord) {
          s.abilityAliases.set(`${name.toLowerCase()}|${lastWord}`, ab);
        }
      }
    }
    store = s;
    return s;
  }

  get isLoaded() {
    return Object.keys(this.champions).length > 0;
  }

  resolveChampionName(input: string): string | undefined {
    const lower = input.trim().toLowerCase();
    if (this.aliasToName.has(lower)) return this.aliasToName.get(lower);
    for (const [name] of Object.entries(this.champions)) {
      const nl = name.toLowerCase();
      if (nl.startsWith(lower) || lower.startsWith(nl)) return name;
    }
    return undefined;
  }

  findChampion(name: string): { name: string; data: ChampionSpells } | undefined {
    const canonical = this.resolveChampionName(name);
    if (!canonical) return undefined;
    const data = this.champions[canonical];
    if (!data) return undefined;
    return { name: canonical, data };
  }

  resolveSummoner(token: string): string | undefined {
    return SUMMONER_ALIASES[token.trim().toLowerCase()];
  }

  resolveAbility(championName: string, token: string): string | undefined {
    const upper = token.trim().toUpperCase();
    if (["Q", "W", "E", "R", "D", "F"].includes(upper)) return upper;
    const sum = this.resolveSummoner(token);
    if (sum) return sum;
    const lower = token.trim().toLowerCase();
    const key = `${championName.toLowerCase()}|${lower}`;
    if (this.abilityAliases.has(key)) return this.abilityAliases.get(key);
    const champ = this.champions[championName];
    if (!champ) return undefined;
    for (const ab of ["Q", "W", "E", "R"] as const) {
      const spell = champ[ab];
      if (spell?.name.toLowerCase().includes(lower)) return ab;
    }
    return undefined;
  }

  getSpell(championName: string, ability: string): SpellData & { abilityKey: string } | undefined {
    const key = ability.toUpperCase();
    if (SUMMONER_SPELLS[key]) {
      return { ...SUMMONER_SPELLS[key], abilityKey: key };
    }
    const champ = this.champions[championName];
    if (!champ) return undefined;
    const spell = champ[key as "Q" | "W" | "E" | "R"];
    if (!spell) return undefined;
    return { ...spell, abilityKey: key };
  }
}

export function getSpellsStore(): SpellsStore | null {
  return store;
}

export function effectiveCooldown(baseCd: number, abilityHaste: number): number {
  if (abilityHaste <= 0) return baseCd;
  return (baseCd * 100) / (100 + abilityHaste);
}
