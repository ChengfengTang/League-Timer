use rusqlite::{params, Connection, OptionalExtension, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpellInfo {
    pub champion_key: String,
    pub champion_name: String,
    pub ability_key: String,
    pub spell_name: String,
    pub cooldowns: Vec<f64>,
    pub cd_trigger: String,
    pub cd_delay_secs: f64,
    pub cd_note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChampionBrief {
    pub ddragon_key: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagChunk {
    pub champion_name: String,
    pub ability_key: Option<String>,
    pub content: String,
    pub score: f64,
}

pub struct DbState {
    pub path: PathBuf,
    pub conn: Mutex<Connection>,
}

impl DbState {
    pub fn new(path: PathBuf) -> SqlResult<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = Connection::open(&path)?;
        let state = Self {
            path,
            conn: Mutex::new(conn),
        };
        state.init_session_schema()?;
        Ok(state)
    }

    /// Prefer the ingested DB at the repo root (`../db/league.db` from src-tauri).
    pub fn resolve_db_path() -> PathBuf {
        let mut candidates: Vec<PathBuf> = Vec::new();

        // Stable in dev: always <repo>/db/league.db regardless of process cwd
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if let Some(repo_root) = manifest_dir.parent() {
            candidates.push(repo_root.join("db").join("league.db"));
        }

        if let Ok(cwd) = std::env::current_dir() {
            candidates.push(cwd.join("db").join("league.db"));
            if let Some(parent) = cwd.parent() {
                candidates.push(parent.join("db").join("league.db"));
            }
        }

        if let Ok(home) = std::env::var("HOME") {
            candidates.push(
                PathBuf::from(home)
                    .join("Library")
                    .join("Application Support")
                    .join("com.jon.league-timer")
                    .join("league.db"),
            );
        }

        for path in &candidates {
            if path.exists() && Self::has_champion_tables(path) {
                eprintln!("[league-timer] Using database: {}", path.display());
                return path.clone();
            }
        }

        // Default: repo db path (ingest target)
        manifest_dir
            .parent()
            .map(|p| p.join("db").join("league.db"))
            .unwrap_or_else(|| PathBuf::from("db/league.db"))
    }

    fn has_champion_tables(path: &Path) -> bool {
        let Ok(conn) = Connection::open(path) else {
            return false;
        };
        conn.query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='champions' LIMIT 1",
            [],
            |_| Ok(()),
        )
        .is_ok()
    }

    fn init_session_schema(&self) -> SqlResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS active_champions (
                id TEXT PRIMARY KEY,
                champion_key TEXT NOT NULL,
                name TEXT NOT NULL,
                level INTEGER NOT NULL DEFAULT 1,
                ability_haste INTEGER NOT NULL DEFAULT 0,
                q_rank INTEGER NOT NULL DEFAULT 1,
                w_rank INTEGER NOT NULL DEFAULT 1,
                e_rank INTEGER NOT NULL DEFAULT 1,
                r_rank INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS active_timers (
                id TEXT PRIMARY KEY,
                champion_id TEXT NOT NULL,
                ability_key TEXT NOT NULL,
                ends_at_ms INTEGER NOT NULL,
                effective_cd REAL NOT NULL,
                base_cd REAL NOT NULL,
                UNIQUE(champion_id, ability_key)
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            "#,
        )?;
        Ok(())
    }

    pub fn has_champion_data(&self) -> SqlResult<bool> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM champions", [], |r| r.get(0))
            .unwrap_or(0);
        Ok(count > 0)
    }

    pub fn list_champion_names(&self) -> SqlResult<Vec<ChampionBrief>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT ddragon_key, name FROM champions ORDER BY name")?;
        let rows = stmt.query_map([], |row| {
            Ok(ChampionBrief {
                ddragon_key: row.get(0)?,
                name: row.get(1)?,
            })
        })?;
        rows.collect()
    }

    pub fn find_champion_key(&self, name: &str) -> SqlResult<Option<ChampionBrief>> {
        let conn = self.conn.lock().unwrap();
        let lower = name.to_lowercase();
        conn.query_row(
            "SELECT ddragon_key, name FROM champions WHERE LOWER(name) = ?1 OR LOWER(ddragon_key) = ?1",
            params![lower],
            |row| {
                Ok(ChampionBrief {
                    ddragon_key: row.get(0)?,
                    name: row.get(1)?,
                })
            },
        )
        .optional()
        .or_else(|_| {
            conn.query_row(
                "SELECT ddragon_key, name FROM champions WHERE LOWER(name) LIKE ?1 LIMIT 1",
                params![format!("%{}%", lower)],
                |row| {
                    Ok(ChampionBrief {
                        ddragon_key: row.get(0)?,
                        name: row.get(1)?,
                    })
                },
            )
            .optional()
        })
    }

    pub fn get_spell(&self, champion_key: &str, ability: &str) -> SqlResult<Option<SpellInfo>> {
        let conn = self.conn.lock().unwrap();
        let ability = ability.to_uppercase();
        conn.query_row(
            r#"SELECT champion_key, champion_name, ability_key, spell_name, cooldowns,
                      cd_trigger, cd_delay_secs, cd_note
               FROM spells WHERE champion_key = ?1 AND ability_key = ?2"#,
            params![champion_key, ability],
            |row| {
                let cds: String = row.get(4)?;
                Ok(SpellInfo {
                    champion_key: row.get(0)?,
                    champion_name: row.get(1)?,
                    ability_key: row.get(2)?,
                    spell_name: row.get(3)?,
                    cooldowns: serde_json::from_str(&cds).unwrap_or(vec![0.0]),
                    cd_trigger: row.get(5)?,
                    cd_delay_secs: row.get(6)?,
                    cd_note: row.get(7)?,
                })
            },
        )
        .optional()
    }

    pub fn rag_search(&self, query: &str, limit: usize) -> SqlResult<Vec<RagChunk>> {
        let conn = self.conn.lock().unwrap();
        let pattern = format!("%{}%", query.to_lowercase());
        let mut stmt = conn.prepare(
            r#"SELECT champion_name, ability_key, content FROM rag_chunks
               WHERE LOWER(content) LIKE ?1 OR LOWER(champion_name) LIKE ?1
               LIMIT ?2"#,
        )?;
        let rows = stmt.query_map(params![pattern, limit as i64], |row| {
            Ok(RagChunk {
                champion_name: row.get(0)?,
                ability_key: row.get(1)?,
                content: row.get(2)?,
                score: 1.0,
            })
        })?;
        rows.collect()
    }

    pub fn rag_search_semantic(&self, query_embedding: &[f32], limit: usize) -> SqlResult<Vec<RagChunk>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT champion_name, ability_key, content, embedding FROM rag_chunks WHERE embedding IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |row| {
            let emb_str: String = row.get(3)?;
            let emb: Vec<f32> = serde_json::from_str(&emb_str).unwrap_or_default();
            let score = cosine_similarity(query_embedding, &emb);
            Ok((
                RagChunk {
                    champion_name: row.get(0)?,
                    ability_key: row.get(1)?,
                    content: row.get(2)?,
                    score,
                },
                score,
            ))
        })?;

        let mut results: Vec<RagChunk> = rows.filter_map(|r| r.ok().map(|(c, _)| c)).collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    }

    pub fn get_meta(&self, key: &str) -> SqlResult<Option<String>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT value FROM meta WHERE key = ?1",
            params![key],
            |row| row.get(0),
        )
        .optional()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

pub fn effective_cooldown(base_cd: f64, ability_haste: i32) -> f64 {
    if ability_haste <= 0 {
        return base_cd;
    }
    base_cd * 100.0 / (100.0 + ability_haste as f64)
}
