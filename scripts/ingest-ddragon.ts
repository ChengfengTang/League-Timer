import "dotenv/config";
import { readFileSync, mkdirSync, writeFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const SPELLS_PATH = join(ROOT, "data", "spells.json");
const PUBLIC_SPELLS_PATH = join(ROOT, "public", "spells.json");
const RULES_PATH = join(ROOT, "data", "cd_rules.json");

type CdRule = {
  champion: string;
  ability: string;
  cd_trigger: string;
  cd_delay_secs?: number;
  note?: string;
};

type SpellData = {
  name: string;
  cooldowns: number[];
  cd_trigger: string;
  cd_delay_secs: number;
};

type ChampionSpells = {
  key: string;
  aliases: string[];
  Q?: SpellData;
  W?: SpellData;
  E?: SpellData;
  R?: SpellData;
};

type DDragonChampion = {
  id: string;
  key: string;
  name: string;
  title: string;
  spells: Array<{
    name: string;
    cooldownBurn: string;
  }>;
};

const ABILITY_KEYS = ["Q", "W", "E", "R"] as const;

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed ${url}: ${res.status}`);
  return res.json() as Promise<T>;
}

async function getLatestVersion(): Promise<string> {
  const pinned = process.env.DDRAGON_VERSION?.trim();
  if (pinned) return pinned;
  const versions = await fetchJson<string[]>(
    "https://ddragon.leagueoflegends.com/api/versions.json",
  );
  return versions[0];
}

function parseCooldowns(cooldownBurn: string): number[] {
  if (!cooldownBurn || cooldownBurn === "0") return [0];
  return cooldownBurn.split("/").map((s) => parseFloat(s.trim()));
}

function ruleKey(champion: string, ability: string) {
  return `${champion.toLowerCase()}|${ability.toUpperCase()}`;
}

async function main() {
  mkdirSync(join(ROOT, "data"), { recursive: true });
  mkdirSync(join(ROOT, "public"), { recursive: true });

  const rules: CdRule[] = JSON.parse(readFileSync(RULES_PATH, "utf-8"));
  const ruleMap = new Map<string, CdRule>();
  for (const r of rules) {
    ruleMap.set(ruleKey(r.champion, r.ability), r);
  }

  const version = await getLatestVersion();
  console.log(`Using Data Dragon version: ${version}`);

  const championList = await fetchJson<{ data: Record<string, { id: string }> }>(
    `https://ddragon.leagueoflegends.com/cdn/${version}/data/en_US/champion.json`,
  );

  const champions: Record<string, ChampionSpells> = {};
  const briefs = Object.values(championList.data);
  let spellCount = 0;

  for (const brief of briefs) {
    const detail = await fetchJson<{ data: Record<string, DDragonChampion> }>(
      `https://ddragon.leagueoflegends.com/cdn/${version}/data/en_US/champion/${brief.id}.json`,
    );
    const champ = detail.data[brief.id];
    const entry: ChampionSpells = {
      key: champ.key,
      aliases: [
        champ.name.toLowerCase(),
        champ.key.toLowerCase(),
        champ.id.toLowerCase(),
      ],
    };

    for (let i = 0; i < ABILITY_KEYS.length; i++) {
      const abilityKey = ABILITY_KEYS[i];
      const spell = champ.spells[i];
      const cooldowns = parseCooldowns(spell.cooldownBurn);
      const rule = ruleMap.get(ruleKey(champ.name, abilityKey));
      const spellData: SpellData = {
        name: spell.name,
        cooldowns,
        cd_trigger: rule?.cd_trigger ?? "on_cast",
        cd_delay_secs: rule?.cd_delay_secs ?? 0,
      };
      entry[abilityKey] = spellData;
      spellCount++;
    }

    champions[champ.name] = entry;
  }

  const output = {
    meta: {
      ddragon_version: version,
      ingested_at: new Date().toISOString(),
    },
    champions,
  };

  const json = JSON.stringify(output, null, 2);
  writeFileSync(SPELLS_PATH, json);
  writeFileSync(PUBLIC_SPELLS_PATH, json);
  console.log(`Wrote ${briefs.length} champions, ${spellCount} spells to`);
  console.log(`  ${SPELLS_PATH}`);
  console.log(`  ${PUBLIC_SPELLS_PATH}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
