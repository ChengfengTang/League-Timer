import { invoke } from "@tauri-apps/api/core";

export async function queryChampion(question: string): Promise<string> {
  return invoke<string>("rag_query", { question });
}
