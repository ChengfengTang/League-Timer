export type TimerStatus = "idle" | "ticking" | "ready";

export interface AbilityState {
  key: string;
  rank: number;
  status: TimerStatus;
  remaining_secs: number;
  effective_cd: number;
  base_cd: number;
  spell_name: string;
}

export interface ActiveChampion {
  id: string;
  champion_key: string;
  name: string;
  level: number;
  ability_haste: number;
  abilities: AbilityState[];
}

export interface AppInfo {
  db_path: string;
  has_data: boolean;
  ddragon_version: string | null;
}

export interface AbilityReadyEvent {
  champion: string;
  ability: string;
  message: string;
}
