# League Timer

Desktop voice-controlled cooldown tracker for League of Legends. Track enemy (or ally) champion abilities, start timers by voice or click, and hear announcements when abilities come back up.

## Features

- Champion cards with **Q / W / E / R** cooldown timers
- **Voice commands** via push-to-talk (`⌘⇧Space` or hold the mic button): e.g. “add champion Ahri”, “Ahri used E”
- **TTS announcements** when an ability is ready: “Ahri E back up”
- Adjust **level**, **ability haste**, and **ability rank** by voice or UI
- **RAG** over ingested Data Dragon + curated cooldown-start rules (on cast, on hit, after channel, etc.)
- **Overlay window** (always on top) for in-game use
- Session **persistence** in SQLite

## Prerequisites

- [Node.js](https://nodejs.org/) 20+
- [Rust](https://www.rust-lang.org/tools/install)
- [Tauri prerequisites](https://tauri.app/start/prerequisites/) for your OS
- OpenAI API key (Whisper, GPT-4o-mini, TTS, embeddings)

## Troubleshooting

**`no such table: champions`** — The app was pointing at the wrong database. Ingest writes to `db/league.db` at the project root. If you see an empty DB under `src-tauri/db/`, delete it and restart. The app now always prefers `<project>/db/league.db`.

## Setup

```bash
cp .env.example .env
# Add OPENAI_API_KEY=your-key-from-platform.openai.com

npm install

# Download champion data (~2 min)
npm run ingest

# Optional: semantic RAG embeddings
npm run embeddings

# Run desktop app
npm run tauri dev
```

## Voice examples

| Say | Effect |
|-----|--------|
| add champion ahri | Adds Ahri to the tracker |
| ahri used E | Starts Ahri E cooldown |
| ahri E landed | Starts CD for on-hit abilities (e.g. Zoe E) |
| ahri level 11 | Sets level |
| 40 ability haste on ahri | Sets ability haste |
| ahri E rank 3 | Sets E rank for base CD lookup |
| When does Zoe E go on cooldown? | RAG answer |

## Data

- **Data Dragon** — base cooldowns per rank (`npm run ingest`)
- **`data/cd_rules.json`** — when cooldowns start (on cast, on hit, after channel, etc.)

Re-ingest after patches:

```bash
npm run ingest
npm run embeddings
```

## Project structure

- `src/` — React UI
- `src-tauri/` — Rust cooldown engine, SQLite, OpenAI integration
- `scripts/ingest-ddragon.ts` — Data Dragon → `db/league.db`
- `scripts/build-embeddings.ts` — RAG vector index

## License

MIT
