import { useCallback, useRef, useState } from "react";
import { useGame } from "../context/GameContext";
import {
  createSpeechListener,
  isSpeechRecognitionSupported,
} from "../lib/speech";

export function useVoice(onLogs?: (logs: string[]) => void) {
  const { processTranscript } = useGame();
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);
  const listenerRef = useRef<ReturnType<typeof createSpeechListener> | null>(
    null,
  );

  const finish = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      setTranscript(trimmed);
      const logs = processTranscript(trimmed);
      onLogs?.(logs);
    },
    [processTranscript, onLogs],
  );

  const startRecording = useCallback(() => {
    if (listenerRef.current?.isActive()) return;
    setError(null);
    if (!isSpeechRecognitionSupported()) {
      setError("Use Chrome or Edge for voice (Web Speech API)");
      return;
    }
    listenerRef.current = createSpeechListener({
      onResult: (text) => setTranscript(text),
      onError: (msg) => setError(msg),
      onEnd: (text) => {
        setRecording(false);
        if (text) finish(text);
      },
    });
    setRecording(true);
    listenerRef.current.start();
  }, [finish]);

  const stopRecording = useCallback(() => {
    listenerRef.current?.stop();
  }, []);

  const processText = useCallback(
    (text: string) => {
      setError(null);
      finish(text);
    },
    [finish],
  );

  return {
    recording,
    processing: false,
    transcript,
    error,
    startRecording,
    stopRecording,
    processText,
  };
}
