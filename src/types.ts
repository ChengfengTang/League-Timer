export type TimerStatus = "idle" | "ticking" | "ready";

export interface SpellData {
  name: string;
  cooldowns: number[];
  cd_trigger: string;
  cd_delay_secs: number;
}

export interface ChampionSpells {
  key: string;
  aliases?: string[];
  Q?: SpellData;
  W?: SpellData;
  E?: SpellData;
  R?: SpellData;
}

export interface SpellsFile {
  meta?: Record<string, string>;
  champions: Record<string, ChampionSpells>;
}

export interface AbilityState {
  key: string;
  rank: number;
  status: TimerStatus;
  remaining_secs: number;
  effective_cd: number;
  base_cd: number;
  spell_name: string;
  ends_at_ms?: number | null;
}

export interface ActiveChampion {
  id: string;
  champion_key: string;
  name: string;
  level: number;
  ability_haste: number;
  abilities: AbilityState[];
}

export interface VoiceAction {
  action: string;
  champion?: string;
  ability?: string;
  level?: number;
  ability_haste?: number;
  rank?: number;
  confirm_hit?: boolean;
}

export interface AbilityReadyEvent {
  champion: string;
  ability: string;
  message: string;
}
