import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { register, unregister } from "@tauri-apps/plugin-global-shortcut";
import { WebviewWindow } from "@tauri-apps/api/webviewWindow";
import { ChampionCard } from "./components/ChampionCard";
import { useCooldownEvents } from "./hooks/useCooldownEvents";
import { useVoice, playTtsBase64 } from "./hooks/useVoice";
import { queryChampion } from "./services/rag";
import type { AbilityReadyEvent, AppInfo } from "./types";

const isOverlay =
  new URLSearchParams(window.location.search).get("overlay") === "1";

function App() {
  const [info, setInfo] = useState<AppInfo | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [addName, setAddName] = useState("");
  const [textCmd, setTextCmd] = useState("");
  const [ragQ, setRagQ] = useState("");
  const [ragA, setRagA] = useState("");

  const onReady = useCallback(
    async (evt: AbilityReadyEvent) => {
      setLogs((l) => [evt.message, ...l].slice(0, 20));
      if (!ttsEnabled) return;
      try {
        const b64 = await invoke<string>("speak_text", { text: evt.message });
        await playTtsBase64(b64);
      } catch {
        /* TTS optional */
      }
    },
    [ttsEnabled],
  );

  const { champions, refresh } = useCooldownEvents(onReady);

  const onVoiceDone = useCallback(
    (voiceLogs: string[]) => {
      setLogs((l) => [...voiceLogs, ...l].slice(0, 20));
      refresh();
    },
    [refresh],
  );

  const voice = useVoice(onVoiceDone);

  useEffect(() => {
    invoke<AppInfo>("get_app_info").then(setInfo);
    invoke<string | null>("get_setting", { key: "tts_enabled" }).then((v) => {
      if (v !== null) setTtsEnabled(v === "true");
    });
  }, []);

  useEffect(() => {
    if (isOverlay) return;
    const shortcut = "CommandOrControl+Shift+Space";
    register(shortcut, () => {
      if (voice.recording) voice.stopRecording();
      else voice.startRecording();
    }).catch(console.error);
    return () => {
      unregister(shortcut).catch(console.error);
    };
  }, [voice.recording, voice.startRecording, voice.stopRecording]);

  const addChampion = async () => {
    if (!addName.trim()) return;
    try {
      await invoke("add_champion", { name: addName.trim() });
      setAddName("");
      refresh();
    } catch (e) {
      setLogs((l) => [String(e), ...l]);
    }
  };

  const toggleOverlay = async () => {
    const overlay = await WebviewWindow.getByLabel("overlay");
    if (overlay) {
      const visible = await overlay.isVisible();
      if (visible) await overlay.hide();
      else {
        await overlay.show();
        await overlay.setFocus();
      }
    }
  };

  const toggleTts = async () => {
    const next = !ttsEnabled;
    setTtsEnabled(next);
    await invoke("set_setting", { key: "tts_enabled", value: String(next) });
  };

  if (isOverlay) {
    return (
      <div className="p-2 bg-slate-950/90 min-h-screen">
        <div className="grid gap-2">
          {champions.map((c) => (
            <ChampionCard
              key={c.id}
              champion={c}
              compact
              onUpdate={refresh}
            />
          ))}
          {champions.length === 0 && (
            <p className="text-xs text-slate-500 text-center">No champions</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-800 px-6 py-4 flex flex-wrap items-center gap-4">
        <div>
          <h1 className="text-xl font-bold text-amber-400">League Timer</h1>
          <p className="text-xs text-slate-500">
            Voice cooldown tracker
            {info?.ddragon_version && ` · Patch ${info.ddragon_version}`}
            {!info?.has_data && " · Run npm run ingest"}
          </p>
        </div>
        <div className="flex-1" />
        <button
          type="button"
          onClick={toggleOverlay}
          className="text-sm px-3 py-1.5 rounded-lg border border-slate-600 hover:bg-slate-800"
        >
          Overlay
        </button>
        <button
          type="button"
          onClick={toggleTts}
          className={`text-sm px-3 py-1.5 rounded-lg border ${
            ttsEnabled
              ? "border-emerald-600 text-emerald-400"
              : "border-slate-600 text-slate-500"
          }`}
        >
          TTS {ttsEnabled ? "On" : "Off"}
        </button>
      </header>

      <main className="flex-1 p-6 grid lg:grid-cols-[1fr_320px] gap-6">
        <section>
          <div className="flex flex-wrap gap-2 mb-4">
            <input
              value={addName}
              onChange={(e) => setAddName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addChampion()}
              placeholder="Add champion (e.g. Ahri)"
              className="flex-1 min-w-[200px] bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm"
            />
            <button
              type="button"
              onClick={addChampion}
              className="px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-sm font-medium"
            >
              Add
            </button>
          </div>

          {champions.length === 0 ? (
            <p className="text-slate-500 text-sm">
              Add champions manually or say &quot;add champion Ahri&quot;. Click
              abilities to test cooldowns. Hold{" "}
              <kbd className="px-1 bg-slate-800 rounded">⌘⇧Space</kbd> for voice.
            </p>
          ) : (
            <div className="grid sm:grid-cols-2 gap-4">
              {champions.map((c) => (
                <ChampionCard key={c.id} champion={c} onUpdate={refresh} />
              ))}
            </div>
          )}
        </section>

        <aside className="space-y-4">
          <div className="rounded-xl border border-slate-700 bg-slate-900/80 p-4">
            <h2 className="text-sm font-semibold mb-3">Voice</h2>
            <button
              type="button"
              onMouseDown={voice.startRecording}
              onMouseUp={voice.stopRecording}
              onTouchStart={(e) => {
                e.preventDefault();
                voice.startRecording();
              }}
              onTouchEnd={voice.stopRecording}
              className={`w-full py-4 rounded-xl font-medium transition ${
                voice.recording
                  ? "bg-red-600 animate-pulse"
                  : "bg-indigo-600 hover:bg-indigo-500"
              }`}
            >
              {voice.recording
                ? "Listening…"
                : voice.processing
                  ? "Processing…"
                  : "Hold to talk"}
            </button>
            <p className="text-[10px] text-slate-500 mt-2">
              Global: ⌘⇧Space · Right-click ability to reset
            </p>
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
                placeholder='Type command: "ahri used E"'
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

          <div className="rounded-xl border border-slate-700 bg-slate-900/80 p-4">
            <h2 className="text-sm font-semibold mb-2">RAG query</h2>
            <input
              value={ragQ}
              onChange={(e) => setRagQ(e.target.value)}
              placeholder="When does Ahri R go on CD?"
              className="w-full text-xs bg-slate-800 border border-slate-600 rounded px-2 py-1.5 mb-2"
            />
            <button
              type="button"
              onClick={async () => {
                setRagA("…");
                const a = await queryChampion(ragQ);
                setRagA(a);
              }}
              className="text-xs px-3 py-1 rounded bg-slate-700 hover:bg-slate-600"
            >
              Ask
            </button>
            {ragA && (
              <p className="text-xs mt-2 text-slate-400 whitespace-pre-wrap">
                {ragA}
              </p>
            )}
          </div>

          <div className="rounded-xl border border-slate-700 bg-slate-900/80 p-4 max-h-48 overflow-y-auto">
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

export default App;
