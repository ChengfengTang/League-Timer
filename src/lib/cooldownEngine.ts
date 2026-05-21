import type { AbilityState, ActiveChampion, VoiceAction } from "../types";
import { effectiveCooldown, SpellsStore } from "./spells";

const SESSION_KEY = "league-timer-session";

type TimerEntry = {
  championId: string;
  championName: string;
  abilityKey: string;
  endsAtMs: number;
  effectiveCd: number;
};

type SessionData = {
  champions: ActiveChampion[];
};

function uuid(): string {
  return crypto.randomUUID();
}

function defaultAbility(
  key: string,
  spellName: string,
): AbilityState {
  return {
    key,
    rank: 1,
    status: "idle",
    remaining_secs: 0,
    effective_cd: 0,
    base_cd: 0,
    spell_name: spellName,
    ends_at_ms: null,
  };
}

export type ReadyCallback = (evt: { champion: string; ability: string; message: string }) => void;

export class CooldownEngine {
  champions = new Map<string, ActiveChampion>();
  timers = new Map<string, TimerEntry>();
  private listeners = new Set<() => void>();
  private onReady?: ReadyCallback;
  private tickId: ReturnType<typeof setInterval> | null = null;

  constructor(
    private spells: SpellsStore,
    onReady?: ReadyCallback,
  ) {
    this.onReady = onReady;
    this.loadSession();
    this.startTick();
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private notify() {
    this.listeners.forEach((fn) => fn());
    this.saveSession();
  }

  getSnapshot(): ActiveChampion[] {
    this.syncRemaining();
    return [...this.champions.values()];
  }

  private syncRemaining() {
    const now = Date.now();
    for (const champ of this.champions.values()) {
      for (const ab of champ.abilities) {
        const tid = `${champ.id}:${ab.key}`;
        const t = this.timers.get(tid);
        if (t) {
          const rem = Math.max(0, (t.endsAtMs - now) / 1000);
          ab.ends_at_ms = t.endsAtMs;
          ab.remaining_secs = rem;
          ab.effective_cd = t.effectiveCd;
          ab.status = rem > 0.05 ? "ticking" : "ready";
        } else if (ab.ends_at_ms != null) {
          ab.ends_at_ms = null;
          ab.remaining_secs = 0;
          if (ab.status === "ticking") ab.status = "idle";
        }
      }
    }
  }

  findChampionId(name: string): string | undefined {
    const lower = name.toLowerCase();
    for (const c of this.champions.values()) {
      if (c.name.toLowerCase() === lower || c.id === name) return c.id;
    }
    return undefined;
  }

  addChampion(name: string): ActiveChampion {
    const found = this.spells.findChampion(name);
    if (!found) throw new Error(`Unknown champion: ${name}. Run npm run ingest.`);
    const { name: champName, data } = found;
    if ([...this.champions.values()].some((c) => c.name === champName)) {
      throw new Error(`${champName} is already tracked`);
    }
    const abilities: AbilityState[] = [];
    for (const ab of ["Q", "W", "E", "R"]) {
      const spell = this.spells.getSpell(champName, ab);
      if (!spell) throw new Error(`Missing ${champName} ${ab}`);
      abilities.push(defaultAbility(ab, spell.name));
    }
    const champ: ActiveChampion = {
      id: uuid(),
      champion_key: data.key,
      name: champName,
      level: 1,
      ability_haste: 0,
      abilities,
    };
    this.champions.set(champ.id, champ);
    this.notify();
    return champ;
  }

  removeChampion(id: string) {
    this.champions.delete(id);
    for (const [tid, t] of this.timers) {
      if (t.championId === id) this.timers.delete(tid);
    }
    this.notify();
  }

  setLevel(id: string, level: number) {
    const c = this.champions.get(id);
    if (!c) throw new Error("Champion not found");
    c.level = Math.min(18, Math.max(1, level));
    this.notify();
  }

  setAbilityHaste(id: string, ah: number) {
    const c = this.champions.get(id);
    if (!c) throw new Error("Champion not found");
    c.ability_haste = Math.max(0, ah);
    this.notify();
  }

  setAbilityRank(id: string, ability: string, rank: number) {
    const c = this.champions.get(id);
    if (!c) throw new Error("Champion not found");
    const ab = c.abilities.find((a) => a.key === ability.toUpperCase());
    if (!ab) throw new Error(`Unknown ability ${ability}`);
    ab.rank = Math.min(5, Math.max(1, rank));
    this.notify();
  }

  abilityUsed(id: string, ability: string, confirmHit = false): ActiveChampion {
    const c = this.champions.get(id);
    if (!c) throw new Error("Champion not found");
    const abKey = ability.toUpperCase();
    const spell = this.spells.getSpell(c.name, abKey);
    if (!spell) throw new Error(`Spell not found: ${c.name} ${abKey}`);

    if (spell.cd_trigger === "on_hit" && !confirmHit) {
      throw new Error(
        `${c.name} ${abKey} CD starts on hit — say "${c.name} ${abKey} landed"`,
      );
    }

    let abState = c.abilities.find((a) => a.key === abKey);
    if (!abState) {
      abState = defaultAbility(abKey, spell.name);
      c.abilities.push(abState);
    }

    const rank = abState.rank;
    const baseCd =
      spell.cooldowns[rank - 1] ?? spell.cooldowns[spell.cooldowns.length - 1] ?? 0;
    const effective = effectiveCooldown(baseCd, c.ability_haste);
    const delayMs = spell.cd_delay_secs * 1000;
    const endsAt = Date.now() + effective * 1000 + delayMs;

    abState.effective_cd = effective;
    abState.base_cd = baseCd;
    abState.spell_name = spell.name;
    abState.ends_at_ms = endsAt;
    abState.remaining_secs = effective + spell.cd_delay_secs;
    abState.status = delayMs > 0 ? "idle" : "ticking";

    this.timers.set(`${id}:${abKey}`, {
      championId: id,
      championName: c.name,
      abilityKey: abKey,
      endsAtMs: endsAt,
      effectiveCd: effective,
    });
    this.notify();
    return c;
  }

  resetAbility(id: string, ability: string) {
    const abKey = ability.toUpperCase();
    this.timers.delete(`${id}:${abKey}`);
    const c = this.champions.get(id);
    if (!c) return;
    const ab = c.abilities.find((a) => a.key === abKey);
    if (ab) {
      ab.status = "idle";
      ab.remaining_secs = 0;
      ab.ends_at_ms = null;
    }
    this.notify();
  }

  applyVoiceAction(act: VoiceAction): string {
    const champ = act.champion ?? "";
    switch (act.action) {
      case "add_champion": {
        const c = this.addChampion(champ);
        return `Added ${c.name}`;
      }
      case "remove_champion": {
        const id = this.findChampionId(champ);
        if (!id) throw new Error(`${champ} not tracked`);
        this.removeChampion(id);
        return `Removed ${champ}`;
      }
      case "ability_used":
      case "ability_hit": {
        const id = this.findChampionId(champ);
        if (!id) throw new Error(`${champ} not tracked`);
        const ability = act.ability;
        if (!ability) throw new Error("Missing ability");
        const confirm =
          act.action === "ability_hit" || act.confirm_hit === true;
        const c = this.abilityUsed(id, ability, confirm);
        const ab = c.abilities.find((a) => a.key === ability.toUpperCase());
        return `${c.name} ${ability} on cooldown (${ab?.remaining_secs.toFixed(1)}s)`;
      }
      case "ability_ready_ack": {
        const id = this.findChampionId(champ);
        if (!id) throw new Error(`${champ} not tracked`);
        const ability = act.ability;
        if (!ability) throw new Error("Missing ability");
        this.resetAbility(id, ability);
        return `${champ} ${ability} marked ready`;
      }
      case "set_level": {
        const id = this.findChampionId(champ);
        if (!id) throw new Error(`${champ} not tracked`);
        this.setLevel(id, act.level ?? 1);
        return `${champ} level ${act.level}`;
      }
      case "set_ability_haste": {
        const id = this.findChampionId(champ);
        if (!id) throw new Error(`${champ} not tracked`);
        this.setAbilityHaste(id, act.ability_haste ?? 0);
        return `${champ} ability haste ${act.ability_haste}`;
      }
      case "set_ability_rank": {
        const id = this.findChampionId(champ);
        if (!id) throw new Error(`${champ} not tracked`);
        const ability = act.ability;
        if (!ability) throw new Error("Missing ability");
        this.setAbilityRank(id, ability, act.rank ?? 1);
        return `${champ} ${ability} rank ${act.rank}`;
      }
      default:
        throw new Error(`Unknown action: ${act.action}`);
    }
  }

  private startTick() {
    if (this.tickId) return;
    this.tickId = setInterval(() => {
      const now = Date.now();
      const ready: TimerEntry[] = [];
      for (const [tid, t] of this.timers) {
        if (t.endsAtMs <= now) {
          ready.push(t);
          this.timers.delete(tid);
        }
      }
      if (ready.length > 0) {
        for (const t of ready) {
          const c = this.champions.get(t.championId);
          if (c) {
            const ab = c.abilities.find((a) => a.key === t.abilityKey);
            if (ab) {
              ab.status = "ready";
              ab.remaining_secs = 0;
              ab.ends_at_ms = null;
            }
          }
          this.onReady?.({
            champion: t.championName,
            ability: t.abilityKey,
            message: `${t.championName} ${t.abilityKey} back up`,
          });
        }
      }
      if (ready.length > 0 || this.champions.size > 0) {
        this.notify();
      }
    }, 50);
  }

  destroy() {
    if (this.tickId) clearInterval(this.tickId);
  }

  private saveSession() {
    const data: SessionData = {
      champions: this.getSnapshot(),
    };
    try {
      localStorage.setItem(SESSION_KEY, JSON.stringify(data));
    } catch {
      /* quota */
    }
  }

  private loadSession() {
    try {
      const raw = localStorage.getItem(SESSION_KEY);
      if (!raw) return;
      const data = JSON.parse(raw) as SessionData;
      const now = Date.now();
      for (const champ of data.champions ?? []) {
        this.champions.set(champ.id, champ);
        for (const ab of champ.abilities) {
          if (ab.ends_at_ms && ab.ends_at_ms > now && ab.status === "ticking") {
            this.timers.set(`${champ.id}:${ab.key}`, {
              championId: champ.id,
              championName: champ.name,
              abilityKey: ab.key,
              endsAtMs: ab.ends_at_ms,
              effectiveCd: ab.effective_cd,
            });
          }
        }
      }
    } catch {
      /* corrupt session */
    }
  }
}
