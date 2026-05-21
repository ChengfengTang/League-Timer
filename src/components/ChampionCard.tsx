import { useGame } from "../context/GameContext";
import type { ActiveChampion } from "../types";
import { AbilityTimer } from "./AbilityTimer";

interface Props {
  champion: ActiveChampion;
  compact?: boolean;
  patchVersion?: string | null;
}

function championImgId(name: string): string {
  return name.replace(/\s+/g, "").replace(/'/g, "");
}

export function ChampionCard({ champion, compact, patchVersion }: Props) {
  const { engine } = useGame();
  const patch = patchVersion ?? "14.24.1";
  const imgSrc = `https://ddragon.leagueoflegends.com/cdn/${patch}/img/champion/${championImgId(champion.name)}.png`;

  if (!engine) return null;

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
          <h3
            className={`font-semibold truncate ${compact ? "text-sm" : "text-lg"}`}
          >
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
                  onChange={(e) => {
                    try {
                      engine.setLevel(champion.id, Number(e.target.value));
                    } catch {
                      /* */
                    }
                  }}
                  className="w-12 bg-slate-800 border border-slate-600 rounded px-1"
                />
              </label>
              <label className="flex items-center gap-1">
                AH
                <input
                  type="number"
                  min={0}
                  value={champion.ability_haste}
                  onChange={(e) => {
                    try {
                      engine.setAbilityHaste(
                        champion.id,
                        Number(e.target.value),
                      );
                    } catch {
                      /* */
                    }
                  }}
                  className="w-12 bg-slate-800 border border-slate-600 rounded px-1"
                />
              </label>
            </div>
          )}
        </div>
        {!compact && (
          <button
            type="button"
            onClick={() => engine.removeChampion(champion.id)}
            className="text-slate-500 hover:text-red-400 text-xs"
          >
            Remove
          </button>
        )}
      </div>
      <div className="flex justify-center gap-2 flex-wrap">
        {champion.abilities.map((ab) => (
          <AbilityTimer
            key={ab.key}
            ability={ab}
            compact={compact}
            onUse={() => {
              try {
                engine.abilityUsed(champion.id, ab.key);
              } catch (e) {
                console.error(e);
              }
            }}
            onReset={() => engine.resetAbility(champion.id, ab.key)}
            onRankChange={(r) => {
              try {
                engine.setAbilityRank(champion.id, ab.key, r);
              } catch (e) {
                console.error(e);
              }
            }}
          />
        ))}
      </div>
    </div>
  );
}
