import { useEffect, useState } from "react";
import type { AbilityState } from "../types";

const RING_R = 15.5;
const RING_C = 2 * Math.PI * RING_R;

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
  const { key, effective_cd, spell_name, rank, ends_at_ms } = ability;
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!ends_at_ms) return;
    const id = window.setInterval(() => setNow(Date.now()), 50);
    return () => window.clearInterval(id);
  }, [ends_at_ms]);

  const remaining =
    ends_at_ms != null
      ? Math.max(0, (ends_at_ms - now) / 1000)
      : ability.remaining_secs;

  const isTicking =
    (ends_at_ms != null && remaining > 0.05) ||
    (ability.status === "ticking" && remaining > 0.05);
  const isReady =
    ability.status === "ready" ||
    (ends_at_ms != null && remaining <= 0.05 && effective_cd > 0);
  const isIdle = !isTicking && !isReady;

  const progress =
    isTicking && effective_cd > 0
      ? Math.min(1, Math.max(0, 1 - remaining / effective_cd))
      : isReady
        ? 1
        : 0;

  const dashOffset = RING_C * (1 - progress);

  const statusColor = isReady
    ? "border-emerald-500 bg-emerald-950/50"
    : isTicking
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
        className={`relative ${size} rounded-full border-2 flex flex-col items-center justify-center transition-colors ${statusColor} hover:brightness-110`}
      >
        <svg
          className="absolute inset-0 -rotate-90"
          viewBox="0 0 36 36"
          aria-hidden
        >
          <circle
            cx="18"
            cy="18"
            r={RING_R}
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="text-slate-700/80"
          />
          {(isTicking || isReady) && (
            <circle
              cx="18"
              cy="18"
              r={RING_R}
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeDasharray={RING_C}
              strokeDashoffset={dashOffset}
              className={`transition-[stroke-dashoffset] duration-75 ease-linear ${
                isReady ? "text-emerald-400" : "text-amber-400"
              }`}
            />
          )}
        </svg>
        <span className="font-bold z-10">{key}</span>
        {isTicking && (
          <span className="z-10 text-[10px] tabular-nums leading-none">
            {remaining >= 10 ? remaining.toFixed(0) : remaining.toFixed(1)}
          </span>
        )}
        {isReady && (
          <span className="z-10 text-[9px] text-emerald-400 font-medium">UP</span>
        )}
        {isIdle && !compact && (
          <span className="z-10 text-[8px] text-slate-500">ready</span>
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
