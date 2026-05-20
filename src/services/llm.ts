import { invoke } from "@tauri-apps/api/core";

export async function processVoiceTranscript(transcript: string): Promise<string[]> {
  return invoke<string[]>("process_voice", { transcript });
}
