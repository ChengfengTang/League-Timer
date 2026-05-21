import { useState } from "react";
import { GameProvider, useGame } from "./context/GameContext";
import { ChampionCard } from "./components/ChampionCard";
import { useVoice } from "./hooks/useVoice";
import { isSpeechRecognitionSupported } from "./lib/speech";

function AppInner() {
  const {
    loading,
    error,
    patchVersion,
    champions,
    ttsEnabled,
    setTtsEnabled,
    addChampion,
  } = useGame();
  const [logs, setLogs] = useState<string[]>([]);
  const [addName, setAddName] = useState("");
  const [textCmd, setTextCmd] = useState("");

  const voice = useVoice((voiceLogs) => {
    setLogs((l) => [...voiceLogs, ...l].slice(0, 20));
  });

  const handleAdd = () => {
    if (!addName.trim()) return;
    try {
      addChampion(addName.trim());
      setAddName("");
    } catch (e) {
      setLogs((l) => [String(e), ...l]);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-slate-400">
        Loading champion data…
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4 p-6 text-center">
        <p className="text-red-400">{error}</p>
        <p className="text-sm text-slate-500">
          Run <code className="bg-slate-800 px-1 rounded">npm run ingest</code> then{" "}
          <code className="bg-slate-800 px-1 rounded">npm run dev</code>
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-800 px-6 py-4 flex flex-wrap items-center gap-4">
        <div>
          <h1 className="text-xl font-bold text-amber-400">League Timer</h1>
          <p className="text-xs text-slate-500">
            Web · regex voice · no API keys
            {patchVersion && ` · Patch ${patchVersion}`}
          </p>
        </div>
        <div className="flex-1" />
        <button
          type="button"
          onClick={() => setTtsEnabled(!ttsEnabled)}
          className={`text-sm px-3 py-1.5 rounded-lg border ${
            ttsEnabled
              ? "border-emerald-600 text-emerald-400"
              : "border-slate-600 text-slate-500"
          }`}
        >
          Voice {ttsEnabled ? "On" : "Off"}
        </button>
      </header>

      <main className="flex-1 p-6 grid lg:grid-cols-[1fr_320px] gap-6">
        <section>
          <div className="flex flex-wrap gap-2 mb-4">
            <input
              value={addName}
              onChange={(e) => setAddName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAdd()}
              placeholder="Add champion (e.g. Ahri)"
              className="flex-1 min-w-[200px] bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm"
            />
            <button
              type="button"
              onClick={handleAdd}
              className="px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-sm font-medium"
            >
              Add
            </button>
          </div>

          {champions.length === 0 ? (
            <p className="text-slate-500 text-sm">
              Add a champion, then say &quot;ahri used E&quot; or hold the mic button.
              Uses browser speech recognition (Chrome/Edge) — instant regex parse.
            </p>
          ) : (
            <div className="grid sm:grid-cols-2 gap-4">
              {champions.map((c) => (
                <ChampionCard
                  key={c.id}
                  champion={c}
                  patchVersion={patchVersion}
                />
              ))}
            </div>
          )}
        </section>

        <aside className="space-y-4">
          <div className="rounded-xl border border-slate-700 bg-slate-900/80 p-4">
            <h2 className="text-sm font-semibold mb-3">Voice</h2>
            {!isSpeechRecognitionSupported() && (
              <p className="text-xs text-amber-500 mb-2">
                Speech recognition needs Chrome or Edge. Type commands below instead.
              </p>
            )}
            <p className="text-xs text-slate-500 mb-2">
              Mic uses Google&apos;s speech service (internet required). Hold the button while you speak, then release.
            </p>
            <button
              type="button"
              onPointerDown={(e) => {
                if (e.button !== 0) return;
                e.currentTarget.setPointerCapture(e.pointerId);
                voice.startRecording();
              }}
              onPointerUp={(e) => {
                e.currentTarget.releasePointerCapture(e.pointerId);
                voice.stopRecording();
              }}
              onPointerCancel={() => voice.stopRecording()}
              onContextMenu={(e) => e.preventDefault()}
              className={`w-full py-4 rounded-xl font-medium transition select-none touch-none ${
                voice.recording
                  ? "bg-red-600 animate-pulse"
                  : "bg-indigo-600 hover:bg-indigo-500"
              }`}
            >
              {voice.recording ? "Listening… release when done" : "Hold to talk"}
            </button>
            {voice.transcript && (
              <p className="text-xs mt-2 text-slate-400 italic">
                &quot;{voice.transcript}&quot;
              </p>
            )}
            {voice.error && (
              <p className="text-xs mt-2 text-red-400">{voice.error}</p>
            )}
            <div className="flex gap-2 mt-3">
              <input
                value={textCmd}
                onChange={(e) => setTextCmd(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && voice.processText(textCmd)}
                placeholder='ahri used E'
                className="flex-1 text-xs bg-slate-800 border border-slate-600 rounded px-2 py-1.5"
              />
              <button
                type="button"
                onClick={() => voice.processText(textCmd)}
                className="text-xs px-2 py-1.5 rounded bg-slate-700 hover:bg-slate-600"
              >
                Run
              </button>
            </div>
          </div>

          <div className="rounded-xl border border-slate-700 bg-slate-900/80 p-4 max-h-64 overflow-y-auto">
            <h2 className="text-sm font-semibold mb-2">Log</h2>
            <ul className="text-xs text-slate-400 space-y-1">
              {logs.map((l, i) => (
                <li key={i}>{l}</li>
              ))}
            </ul>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <GameProvider>
      <AppInner />
    </GameProvider>
  );
}
