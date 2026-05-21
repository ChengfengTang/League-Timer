/** Browser Web Speech API — STT in Chrome/Edge sends audio to Google (internet required). */

export function speakAnnouncement(text: string): boolean {
  if (!("speechSynthesis" in window)) return false;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.05;
  window.speechSynthesis.speak(u);
  return true;
}

type SpeechRecognitionCtor = new () => SpeechRecognition;

function getRecognitionCtor(): SpeechRecognitionCtor | null {
  const w = window as Window & {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export function isSpeechRecognitionSupported(): boolean {
  return getRecognitionCtor() !== null;
}

function friendlyError(code: string): string | null {
  switch (code) {
    case "aborted":
      return null;
    case "no-speech":
      return "No speech heard — speak while the button is red, then release";
    case "network":
      return "Cannot reach speech service (Chrome uses Google’s servers — check internet, VPN, or firewall; you can type commands below)";
    case "not-allowed":
      return "Microphone blocked — allow mic access for this site";
    case "audio-capture":
      return "No microphone found";
    case "service-not-allowed":
      return "Speech not allowed on this page (use https:// or localhost)";
    default:
      return code ? `Speech error: ${code}` : null;
  }
}

export type SpeechListenOptions = {
  onResult: (transcript: string, isFinal: boolean) => void;
  onError?: (message: string) => void;
  onEnd?: (finalTranscript: string) => void;
};

export function createSpeechListener(options: SpeechListenOptions): {
  start: () => void;
  stop: () => void;
  isActive: () => boolean;
} {
  const Ctor = getRecognitionCtor();
  if (!Ctor) {
    return {
      start: () =>
        options.onError?.(
          "Speech recognition not supported — use Chrome/Edge or type commands below",
        ),
      stop: () => {},
      isActive: () => false,
    };
  }

  let recognition: SpeechRecognition | null = null;
  let active = false;
  let finalText = "";

  const start = () => {
    if (active) return;
    finalText = "";
    recognition = new Ctor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalText += t;
        else interim += t;
      }
      const combined = (finalText + interim).trim();
      if (!combined) return;
      const isFinal =
        event.results[event.results.length - 1]?.isFinal ?? false;
      options.onResult(combined, isFinal);
    };

    recognition.onerror = (event) => {
      if (event.error === "aborted") return;
      const msg = friendlyError(event.error);
      if (msg) options.onError?.(msg);
    };

    recognition.onend = () => {
      active = false;
      options.onEnd?.(finalText.trim());
      recognition = null;
    };

    try {
      recognition.start();
      active = true;
    } catch {
      active = false;
      options.onError?.("Could not start mic — wait a moment and try again");
    }
  };

  const stop = () => {
    if (!recognition || !active) return;
    try {
      recognition.stop();
    } catch {
      active = false;
    }
  };

  return { start, stop, isActive: () => active };
}
