import { useCallback, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const data = reader.result as string;
      const base64 = data.split(",")[1] ?? "";
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export function useVoice(onProcessed?: (logs: string[]) => void) {
  const [recording, setRecording] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        if (blob.size < 100) {
          setError("No audio captured");
          return;
        }
        setProcessing(true);
        try {
          const b64 = await blobToBase64(blob);
          const text = await invoke<string>("transcribe_voice", {
            audioBase64: b64,
            mime: "audio/webm",
          });
          setTranscript(text);
          const logs = await invoke<string[]>("process_voice", { transcript: text });
          onProcessed?.(logs);
        } catch (e) {
          setError(String(e));
        } finally {
          setProcessing(false);
        }
      };
      mediaRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch (e) {
      setError(String(e));
    }
  }, [onProcessed]);

  const stopRecording = useCallback(() => {
    mediaRef.current?.stop();
    setRecording(false);
  }, []);

  const processText = useCallback(
    async (text: string) => {
      setProcessing(true);
      setError(null);
      setTranscript(text);
      try {
        const logs = await invoke<string[]>("process_voice", { transcript: text });
        onProcessed?.(logs);
      } catch (e) {
        setError(String(e));
      } finally {
        setProcessing(false);
      }
    },
    [onProcessed],
  );

  return {
    recording,
    processing,
    transcript,
    error,
    startRecording,
    stopRecording,
    processText,
  };
}

export async function playTtsBase64(b64: string) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], { type: "audio/mpeg" });
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  await audio.play();
  audio.onended = () => URL.revokeObjectURL(url);
}
