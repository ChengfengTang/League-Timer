# League Timer (Web)

Fast League of Legends cooldown tracker in the browser. **No OpenAI, no API keys, no desktop app** — browser speech recognition + regex parsing + local timers.

## Stack

- **Vite + React** — runs in Chrome or Edge
- **Web Speech API** — speech-to-text on-device (network only for Google’s STT backend in Chrome)
- **Regex parser** — instant command matching
- **`public/spells.json`** — champion CDs from Data Dragon (`npm run ingest`)
- **localStorage** — session persistence
- **speechSynthesis** — “Ahri E back up” announcements (instant, no API)

## Prerequisites

- [Node.js](https://nodejs.org/) 20+
- **Chrome or Edge** (recommended for voice)

## Setup

```bash
npm install
npm run ingest   # downloads CDs → data/spells.json + public/spells.json
npm run dev      # http://localhost:5173
```

Production build:

```bash
npm run build
npm run preview
```

## Optional `.env`

Only used by the ingest script:

```bash
cp .env.example .env
# DDRAGON_VERSION=16.10.1   # optional patch pin
```

No API keys required to run the app.

## Voice commands (regex)

Case-insensitive. Add the champion first: `add champion ahri`.

### Level & ability haste

| Say | Effect |
|-----|--------|
| `ahri level 11` | Set level |
| `ahri ability haste 40` | Set ability haste |
| `40 ability haste on ahri` | Same (alternate order) |

### Abilities Q / W / E / R

`used` and `no` both **start** the cooldown timer.

| Say | Effect |
|-----|--------|
| `ahri used E` | Start E CD |
| `ahri no W` | Start W CD |
| `ahri E down` | Start E CD |
| `ahri charm down` | Start by ability name |
| `ahri E landed` | On-hit CD (e.g. Zoe E) |
| `ahri E rank 3` | Set rank for base CD |
| `ahri E back up` | Clear timer / mark ready |

### Summoner spells

| Say | Effect |
|-----|--------|
| `ahri used flash` | Flash CD (300s default) |
| `ahri no ignite` | Ignite CD (180s) |

Words: `flash`, `ignite`, `teleport` / `tp`, `ghost`, `heal`, `barrier`, `exhaust`, `cleanse`, `smite`.

### Regex patterns (reference)

```text
^add champion (.+)$
^remove (.+)$
^(.+?) used ([QWER]|flash|ignite|teleport|tp|ghost|heal|barrier|exhaust|cleanse|smite)$
^(.+?) no ([QWER]|flash|…)$
^(.+?) ([QWER]) landed$
^(.+?) ([QWER]) down$
^(.+?) level (\d+)$
^(.+?) ability haste (\d+)$
^(\d+) ability haste(?: on)? (.+)$
^(.+?) ([QWER]) rank (\d+)$
^(.+?) ([QWER]) back up$
^(.+?) (.+?) down$          # ability name, e.g. "ahri charm down"
```

Type the same phrases in the **Run** box to test without the mic.

## Mic tips

- Allow microphone permission when prompted
- Hold **Hold to talk** — release when done; parsing runs on final transcript
- Log shows `[parsed: regex]` on success

## Data

```bash
npm run ingest   # refresh after patches
```

- [`data/spells.json`](data/spells.json) — source copy
- [`public/spells.json`](public/spells.json) — served to the app
- [`data/cd_rules.json`](data/cd_rules.json) — CD start rules merged at ingest

## License

MIT
