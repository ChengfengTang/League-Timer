import type { SpellData } from "../types";

export const SUMMONER_SPELLS: Record<string, SpellData> = {
  FLASH: { name: "Flash", cooldowns: [300], cd_trigger: "on_cast", cd_delay_secs: 0 },
  IGNITE: { name: "Ignite", cooldowns: [180], cd_trigger: "on_cast", cd_delay_secs: 0 },
  TELEPORT: { name: "Teleport", cooldowns: [420], cd_trigger: "on_cast", cd_delay_secs: 0 },
  GHOST: { name: "Ghost", cooldowns: [180], cd_trigger: "on_cast", cd_delay_secs: 0 },
  HEAL: { name: "Heal", cooldowns: [240], cd_trigger: "on_cast", cd_delay_secs: 0 },
  BARRIER: { name: "Barrier", cooldowns: [180], cd_trigger: "on_cast", cd_delay_secs: 0 },
  EXHAUST: { name: "Exhaust", cooldowns: [210], cd_trigger: "on_cast", cd_delay_secs: 0 },
  CLEANSE: { name: "Cleanse", cooldowns: [210], cd_trigger: "on_cast", cd_delay_secs: 0 },
  SMITE: { name: "Smite", cooldowns: [90], cd_trigger: "on_cast", cd_delay_secs: 0 },
};

export const SUMMONER_ALIASES: Record<string, string> = {
  flash: "FLASH",
  ignite: "IGNITE",
  teleport: "TELEPORT",
  tp: "TELEPORT",
  ghost: "GHOST",
  heal: "HEAL",
  barrier: "BARRIER",
  exhaust: "EXHAUST",
  cleanse: "CLEANSE",
  smite: "SMITE",
};
