import { invoke } from "@tauri-apps/api/core";
import type { ActiveChampion } from "../types";
import { AbilityTimer } from "./AbilityTimer";

const DDRAGON =
  "https://ddragon.leagueoflegends.com/cdn/14.24.1/img/champion";

interface Props {
  champion: ActiveChampion;
  compact?: boolean;
  onUpdate: () => void;
}

export function ChampionCard({ champion, compact, onUpdate }: Props) {
  const imgSrc = `${DDRAGON}/${champion.champion_key}.png`;

  const useAbility = async (ability: string) => {
    await invoke("ability_used", {
      championId: champion.id,
      ability,
      confirmHit: false,
    });
    onUpdate();
  };

  const resetAbility = async (ability: string) => {
    await invoke("reset_ability", { championId: champion.id, ability });
    onUpdate();
  };

  const setRank = async (ability: string, rank: number) => {
    await invoke("set_ability_rank", {
      championId: champion.id,
      ability,
      rank,
    });
    onUpdate();
  };

  const setLevel = async (level: number) => {
    await invoke("set_level", { championId: champion.id, level });
    onUpdate();
  };

  const setAh = async (ah: number) => {
    await invoke("set_ability_haste", {
      championId: champion.id,
      abilityHaste: ah,
    });
    onUpdate();
  };

  const remove = async () => {
    await invoke("remove_champion", { championId: champion.id });
    onUpdate();
  };

  return (
    <div
      className={`rounded-xl border border-slate-700 bg-slate-900/80 ${
        compact ? "p-2" : "p-4"
      }`}
    >
      <div className={`flex items-center gap-3 ${compact ? "mb-2" : "mb-4"}`}>
        <img
          src={imgSrc}
          alt={champion.name}
          className={`rounded-lg object-cover bg-slate-800 ${
            compact ? "w-10 h-10" : "w-14 h-14"
          }`}
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
        <div className="flex-1 min-w-0">
          <h3 className={`font-semibold truncate ${compact ? "text-sm" : "text-lg"}`}>
            {champion.name}
          </h3>
          {!compact && (
            <div className="flex flex-wrap gap-2 mt-1 text-xs text-slate-400">
              <label className="flex items-center gap-1">
                Lvl
                <input
                  type="number"
                  min={1}
                  max={18}
                  value={champion.level}
                  onChange={(e) => setLevel(Number(e.target.value))}
                  className="w-12 bg-slate-800 border border-slate-600 rounded px-1"
                />
              </label>
              <label className="flex items-center gap-1">
                AH
                <input
                  type="number"
                  min={0}
                  value={champion.ability_haste}
                  onChange={(e) => setAh(Number(e.target.value))}
                  className="w-12 bg-slate-800 border border-slate-600 rounded px-1"
                />
              </label>
            </div>
          )}
        </div>
        {!compact && (
          <button
            type="button"
            onClick={remove}
            className="text-slate-500 hover:text-red-400 text-xs"
          >
            Remove
          </button>
        )}
      </div>
      <div className="flex justify-center gap-2">
        {champion.abilities.map((ab) => (
          <AbilityTimer
            key={ab.key}
            ability={ab}
            compact={compact}
            onUse={() => useAbility(ab.key)}
            onReset={() => resetAbility(ab.key)}
            onRankChange={(r) => setRank(ab.key, r)}
          />
        ))}
      </div>
    </div>
  );
}
