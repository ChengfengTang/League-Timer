import "dotenv/config";
import Database from "better-sqlite3";
import { readFileSync, mkdirSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const DB_PATH = join(ROOT, "db", "league.db");
const RULES_PATH = join(ROOT, "data", "cd_rules.json");

type CdRule = {
  champion: string;
  ability: string;
  cd_trigger: string;
  cd_delay_secs?: number;
  note?: string;
};

type DDragonChampion = {
  id: string;
  key: string;
  name: string;
  title: string;
  spells: Array<{
    id: string;
    name: string;
    description: string;
    cooldownBurn: string;
    costBurn: string;
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
  mkdirSync(join(ROOT, "db"), { recursive: true });
  if (existsSync(DB_PATH)) {
    console.log(`Removing existing DB at ${DB_PATH}`);
  }
  const db = new Database(DB_PATH);

  db.exec(`
    CREATE TABLE champions (
      ddragon_key TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      title TEXT
    );
    CREATE TABLE spells (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      champion_key TEXT NOT NULL,
      champion_name TEXT NOT NULL,
      ability_key TEXT NOT NULL,
      spell_name TEXT NOT NULL,
      description TEXT,
      cooldowns TEXT NOT NULL,
      cost TEXT,
      cd_trigger TEXT NOT NULL DEFAULT 'on_cast',
      cd_delay_secs REAL NOT NULL DEFAULT 0,
      cd_note TEXT,
      UNIQUE(champion_key, ability_key)
    );
    CREATE TABLE rag_chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      champion_key TEXT NOT NULL,
      champion_name TEXT NOT NULL,
      ability_key TEXT,
      content TEXT NOT NULL,
      embedding TEXT
    );
    CREATE TABLE meta (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
  `);

  const rules: CdRule[] = JSON.parse(readFileSync(RULES_PATH, "utf-8"));
  const ruleMap = new Map<string, CdRule>();
  for (const r of rules) {
    ruleMap.set(ruleKey(r.champion, r.ability), r);
  }

  const version = await getLatestVersion();
  console.log(`Using Data Dragon version: ${version}`);

  const championList = await fetchJson<{ data: Record<string, { key: string; name: string }> }>(
    `https://ddragon.leagueoflegends.com/cdn/${version}/data/en_US/champion.json`,
  );

  const insertChampion = db.prepare(
    `INSERT OR REPLACE INTO champions (ddragon_key, name, title) VALUES (?, ?, ?)`,
  );
  const insertSpell = db.prepare(`
    INSERT OR REPLACE INTO spells
      (champion_key, champion_name, ability_key, spell_name, description, cooldowns, cost, cd_trigger, cd_delay_secs, cd_note)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const insertChunk = db.prepare(`
    INSERT INTO rag_chunks (champion_key, champion_name, ability_key, content)
    VALUES (?, ?, ?, ?)
  `);

  const champions = Object.values(championList.data);
  let spellCount = 0;

  for (const brief of champions) {
    const detail = await fetchJson<{ data: Record<string, DDragonChampion> }>(
      `https://ddragon.leagueoflegends.com/cdn/${version}/data/en_US/champion/${brief.id}.json`,
    );
    const champ = detail.data[brief.id];
    insertChampion.run(champ.key, champ.name, champ.title);

    for (let i = 0; i < ABILITY_KEYS.length; i++) {
      const abilityKey = ABILITY_KEYS[i];
      const spell = champ.spells[i];
      const cooldowns = parseCooldowns(spell.cooldownBurn);
      const rule = ruleMap.get(ruleKey(champ.name, abilityKey));
      const cdTrigger = rule?.cd_trigger ?? "on_cast";
      const cdDelay = rule?.cd_delay_secs ?? 0;
      const cdNote = rule?.note ?? null;

      insertSpell.run(
        champ.key,
        champ.name,
        abilityKey,
        spell.name,
        spell.description.replace(/<[^>]+>/g, " ").slice(0, 500),
        JSON.stringify(cooldowns),
        spell.costBurn || "",
        cdTrigger,
        cdDelay,
        cdNote,
      );
      spellCount++;

      const cdList = cooldowns.join(", ");
      const content = [
        `Champion: ${champ.name} (${champ.title})`,
        `Ability: ${abilityKey} — ${spell.name}`,
        `Base cooldowns by rank (seconds): ${cdList}`,
        `Cooldown starts: ${cdTrigger}${cdDelay > 0 ? ` (delay ${cdDelay}s)` : ""}`,
        cdNote ? `Note: ${cdNote}` : "",
        `Cost: ${spell.costBurn || "none"}`,
        spell.description.replace(/<[^>]+>/g, " ").slice(0, 300),
      ]
        .filter(Boolean)
        .join("\n");

      insertChunk.run(champ.key, champ.name, abilityKey, content);
    }
  }

  db.prepare(`INSERT INTO meta (key, value) VALUES ('ddragon_version', ?)`).run(version);
  db.prepare(`INSERT INTO meta (key, value) VALUES ('ingested_at', ?)`).run(
    new Date().toISOString(),
  );

  console.log(`Ingested ${champions.length} champions, ${spellCount} spells.`);
  console.log(`Database: ${DB_PATH}`);
  console.log(`Run "npm run embeddings" to build RAG vectors (requires OPENAI_API_KEY).`);
  db.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
