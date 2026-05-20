import { useEffect, useState, useCallback } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import type { ActiveChampion, AbilityReadyEvent } from "../types";

export function useCooldownEvents(onReady?: (evt: AbilityReadyEvent) => void) {
  const [champions, setChampions] = useState<ActiveChampion[]>([]);

  const refresh = useCallback(async () => {
    const list = await invoke<ActiveChampion[]>("get_active_champions");
    setChampions(list);
  }, []);

  useEffect(() => {
    refresh();
    const unsubs: Array<() => void> = [];

    listen<ActiveChampion[]>("cooldown-tick", (e) => {
      setChampions(e.payload);
    }).then((unlisten) => unsubs.push(unlisten));

    listen<AbilityReadyEvent>("ability-ready", (e) => {
      onReady?.(e.payload);
    }).then((unlisten) => unsubs.push(unlisten));

    return () => unsubs.forEach((fn) => fn());
  }, [refresh, onReady]);

  return { champions, refresh, setChampions };
}
