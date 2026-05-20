import type { AbilityState } from "../types";

interface Props {
  ability: AbilityState;
  compact?: boolean;
  onUse?: () => void;
  onReset?: () => void;
  onRankChange?: (rank: number) => void;
}

export function AbilityTimer({
  ability,
  compact,
  onUse,
  onReset,
  onRankChange,
}: Props) {
  const { key, status, remaining_secs, effective_cd, spell_name, rank } = ability;
  const progress =
    effective_cd > 0 && status === "ticking"
      ? Math.min(1, 1 - remaining_secs / effective_cd)
      : status === "ready"
        ? 1
        : 0;

  const statusColor =
    status === "ready"
      ? "border-emerald-500 bg-emerald-950/50"
      : status === "ticking"
        ? "border-amber-500 bg-amber-950/40"
        : "border-slate-600 bg-slate-900/60";

  const size = compact ? "w-14 h-14 text-xs" : "w-20 h-20 text-sm";

  return (
    <div className={`flex flex-col items-center gap-1 ${compact ? "scale-90" : ""}`}>
      <button
        type="button"
        title={`${spell_name} — click to start CD`}
        onClick={onUse}
        onContextMenu={(e) => {
          e.preventDefault();
          onReset?.();
        }}
        className={`relative ${size} rounded-full border-2 flex flex-col items-center justify-center transition ${statusColor} hover:brightness-110`}
      >
        <svg className="absolute inset-0 -rotate-90" viewBox="0 0 36 36">
          <circle
            cx="18"
            cy="18"
            r="15.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="text-slate-700"
          />
          <circle
            cx="18"
            cy="18"
            r="15.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeDasharray={`${progress * 97.4} 97.4`}
            className={
              status === "ready" ? "text-emerald-400" : "text-amber-400"
            }
          />
        </svg>
        <span className="font-bold z-10">{key}</span>
        {status === "ticking" && (
          <span className="z-10 text-[10px] tabular-nums">
            {remaining_secs.toFixed(1)}
          </span>
        )}
        {status === "ready" && (
          <span className="z-10 text-[9px] text-emerald-400 font-medium">UP</span>
        )}
      </button>
      {!compact && (
        <select
          className="text-[10px] bg-slate-800 border border-slate-600 rounded px-1 py-0.5"
          value={rank}
          onChange={(e) => onRankChange?.(Number(e.target.value))}
          onClick={(e) => e.stopPropagation()}
        >
          {[1, 2, 3, 4, 5].map((r) => (
            <option key={r} value={r}>
              R{r}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
