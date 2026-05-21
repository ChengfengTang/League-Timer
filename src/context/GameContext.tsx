import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { CooldownEngine } from "../lib/cooldownEngine";
import { parseVoiceRegex } from "../lib/voiceParser";
import { SpellsStore } from "../lib/spells";
import { speakAnnouncement } from "../lib/speech";
import type { ActiveChampion, VoiceAction } from "../types";

type GameContextValue = {
  loading: boolean;
  error: string | null;
  patchVersion: string | null;
  champions: ActiveChampion[];
  ttsEnabled: boolean;
  setTtsEnabled: (v: boolean) => void;
  addChampion: (name: string) => void;
  processTranscript: (text: string) => string[];
  engine: CooldownEngine | null;
};

const GameContext = createContext<GameContextValue | null>(null);

export function GameProvider({ children }: { children: ReactNode }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [spells, setSpells] = useState<SpellsStore | null>(null);
  const [engine, setEngine] = useState<CooldownEngine | null>(null);
  const [champions, setChampions] = useState<ActiveChampion[]>([]);
  const [ttsEnabled, setTtsEnabledState] = useState(() => {
    return localStorage.getItem("tts_enabled") !== "false";
  });

  const setTtsEnabled = useCallback((v: boolean) => {
    setTtsEnabledState(v);
    localStorage.setItem("tts_enabled", String(v));
  }, []);

  const ttsRef = useRef(ttsEnabled);
  ttsRef.current = ttsEnabled;

  useEffect(() => {
    let eng: CooldownEngine | null = null;
    SpellsStore.load()
      .then((s) => {
        setSpells(s);
        eng = new CooldownEngine(s, (evt) => {
          if (!ttsRef.current) return;
          speakAnnouncement(evt.message);
        });
        setEngine(eng);
        setChampions(eng.getSnapshot());
        eng.subscribe(() => setChampions(eng!.getSnapshot()));
        setLoading(false);
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
    return () => eng?.destroy();
  }, []);

  const addChampion = useCallback(
    (name: string) => {
      if (!engine) return;
      try {
        engine.addChampion(name);
      } catch (e) {
        throw e;
      }
    },
    [engine],
  );

  const processTranscript = useCallback(
    (text: string): string[] => {
      if (!engine || !spells) return ["Spells not loaded"];
      const actions = parseVoiceRegex(text, spells);
      if (!actions?.length) {
        return ['[parsed: none] No match — try "ahri used E"'];
      }
      const logs = ["[parsed: regex]"];
      for (const act of actions) {
        try {
          logs.push(engine.applyVoiceAction(act));
        } catch (e) {
          logs.push(`Error: ${e}`);
        }
      }
      return logs;
    },
    [engine, spells],
  );

  const value = useMemo(
    () => ({
      loading,
      error,
      patchVersion: spells?.meta?.ddragon_version ?? null,
      champions,
      ttsEnabled,
      setTtsEnabled,
      addChampion,
      processTranscript,
      engine,
    }),
    [
      loading,
      error,
      spells,
      champions,
      ttsEnabled,
      setTtsEnabled,
      addChampion,
      processTranscript,
      engine,
    ],
  );

  return <GameContext.Provider value={value}>{children}</GameContext.Provider>;
}

export function useGame() {
  const ctx = useContext(GameContext);
  if (!ctx) throw new Error("useGame must be used within GameProvider");
  return ctx;
}

// re-export for typing
export type { VoiceAction };
