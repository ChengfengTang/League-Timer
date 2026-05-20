import "dotenv/config";
import Database from "better-sqlite3";
import OpenAI from "openai";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DB_PATH = join(__dirname, "..", "db", "league.db");

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function embedBatch(texts: string[]): Promise<number[][]> {
  const res = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: texts,
  });
  return res.data.sort((a, b) => a.index - b.index).map((d) => d.embedding);
}

async function main() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY required for embeddings");
  }

  const db = new Database(DB_PATH);
  const rows = db
    .prepare(`SELECT id, content FROM rag_chunks WHERE embedding IS NULL`)
    .all() as { id: number; content: string }[];

  if (rows.length === 0) {
    console.log("All chunks already have embeddings.");
    db.close();
    return;
  }

  const update = db.prepare(`UPDATE rag_chunks SET embedding = ? WHERE id = ?`);
  const BATCH = 50;

  for (let i = 0; i < rows.length; i += BATCH) {
    const batch = rows.slice(i, i + BATCH);
    const vectors = await embedBatch(batch.map((r) => r.content));
    for (let j = 0; j < batch.length; j++) {
      update.run(JSON.stringify(vectors[j]), batch[j].id);
    }
    console.log(`Embedded ${Math.min(i + BATCH, rows.length)} / ${rows.length}`);
  }

  db.close();
  console.log("Embeddings complete.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
