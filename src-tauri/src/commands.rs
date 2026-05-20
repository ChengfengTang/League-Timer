use crate::cooldown::CooldownEngine;
use crate::db::{DbState, RagChunk};
use crate::openai::{self, VoiceAction};
use rusqlite::OptionalExtension;
use serde::Serialize;
use std::sync::Arc;
use tauri::State;

pub struct AppState {
    pub db: Arc<DbState>,
    pub engine: Arc<CooldownEngine>,
}

#[derive(Serialize)]
pub struct AppInfo {
    pub db_path: String,
    pub has_data: bool,
    pub ddragon_version: Option<String>,
}

#[tauri::command]
pub fn get_app_info(state: State<'_, AppState>) -> Result<AppInfo, String> {
    Ok(AppInfo {
        db_path: state.db.path.display().to_string(),
        has_data: state.db.has_champion_data().map_err(|e| e.to_string())?,
        ddragon_version: state.db.get_meta("ddragon_version").map_err(|e| e.to_string())?,
    })
}

#[tauri::command]
pub fn list_champions(state: State<'_, AppState>) -> Result<Vec<crate::db::ChampionBrief>, String> {
    state.db.list_champion_names().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_active_champions(state: State<'_, AppState>) -> Result<Vec<crate::cooldown::ActiveChampion>, String> {
    Ok(state.engine.snapshot())
}

#[tauri::command]
pub fn add_champion(state: State<'_, AppState>, name: String) -> Result<crate::cooldown::ActiveChampion, String> {
    state.engine.add_champion(&name)
}

#[tauri::command]
pub fn remove_champion(state: State<'_, AppState>, champion_id: String) -> Result<(), String> {
    state.engine.remove_champion(&champion_id)
}

#[tauri::command]
pub fn ability_used(
    state: State<'_, AppState>,
    champion_id: String,
    ability: String,
    confirm_hit: Option<bool>,
) -> Result<crate::cooldown::ActiveChampion, String> {
    let (champ, _, _) = state.engine.ability_used(
        &champion_id,
        &ability,
        confirm_hit.unwrap_or(false),
    )?;
    Ok(champ)
}

#[tauri::command]
pub fn ability_used_by_name(
    state: State<'_, AppState>,
    champion_name: String,
    ability: String,
    confirm_hit: Option<bool>,
) -> Result<crate::cooldown::ActiveChampion, String> {
    let id = state
        .engine
        .find_champion_id(&champion_name)
        .ok_or_else(|| format!("{} is not in your tracker", champion_name))?;
    let (champ, _, _) = state
        .engine
        .ability_used(&id, &ability, confirm_hit.unwrap_or(false))?;
    Ok(champ)
}

#[tauri::command]
pub fn reset_ability(
    state: State<'_, AppState>,
    champion_id: String,
    ability: String,
) -> Result<(), String> {
    state.engine.reset_ability(&champion_id, &ability)
}

#[tauri::command]
pub fn set_level(state: State<'_, AppState>, champion_id: String, level: u8) -> Result<(), String> {
    state.engine.set_level(&champion_id, level)
}

#[tauri::command]
pub fn set_ability_haste(
    state: State<'_, AppState>,
    champion_id: String,
    ability_haste: i32,
) -> Result<(), String> {
    state.engine.set_ability_haste(&champion_id, ability_haste)
}

#[tauri::command]
pub fn set_ability_rank(
    state: State<'_, AppState>,
    champion_id: String,
    ability: String,
    rank: u8,
) -> Result<(), String> {
    state.engine.set_ability_rank(&champion_id, &ability, rank)
}

#[tauri::command]
pub async fn transcribe_voice(audio_base64: String, mime: Option<String>) -> Result<String, String> {
    openai::transcribe_audio(&audio_base64, mime.as_deref().unwrap_or("audio/webm")).await
}

#[tauri::command]
pub async fn process_voice(
    state: State<'_, AppState>,
    transcript: String,
) -> Result<Vec<String>, String> {
    let active: Vec<String> = state
        .engine
        .snapshot()
        .into_iter()
        .map(|c| c.name)
        .collect();
    let actions = openai::parse_voice_command(&transcript, &active).await?;
    let mut logs = Vec::new();
    for act in actions {
        match apply_action(&state, act).await {
            Ok(msg) => logs.push(msg),
            Err(e) => logs.push(format!("Error: {}", e)),
        }
    }
    Ok(logs)
}

async fn apply_action(state: &AppState, act: VoiceAction) -> Result<String, String> {
    let champ_name = act.champion.clone().unwrap_or_default();
    match act.action.as_str() {
        "add_champion" => {
            let c = state.engine.add_champion(&champ_name)?;
            Ok(format!("Added {}", c.name))
        }
        "remove_champion" => {
            let id = state
                .engine
                .find_champion_id(&champ_name)
                .ok_or_else(|| format!("{} not tracked", champ_name))?;
            state.engine.remove_champion(&id)?;
            Ok(format!("Removed {}", champ_name))
        }
        "ability_used" | "ability_hit" => {
            let id = state
                .engine
                .find_champion_id(&champ_name)
                .ok_or_else(|| format!("{} not tracked", champ_name))?;
            let ability = act.ability.ok_or("Missing ability")?;
            let confirm = act.action == "ability_hit" || act.confirm_hit.unwrap_or(false);
            let (c, cd, trigger) = state.engine.ability_used(&id, &ability, confirm)?;
            Ok(format!(
                "{} {} on cooldown ({:.1}s, {})",
                c.name, ability, cd, trigger
            ))
        }
        "ability_ready_ack" => {
            let id = state
                .engine
                .find_champion_id(&champ_name)
                .ok_or_else(|| format!("{} not tracked", champ_name))?;
            let ability = act.ability.ok_or("Missing ability")?;
            state.engine.reset_ability(&id, &ability)?;
            Ok(format!("{} {} marked ready", champ_name, ability))
        }
        "set_level" => {
            let id = state.engine.find_champion_id(&champ_name).ok_or("Not tracked")?;
            let level = act.level.ok_or("Missing level")?;
            state.engine.set_level(&id, level)?;
            Ok(format!("{} level {}", champ_name, level))
        }
        "set_ability_haste" => {
            let id = state.engine.find_champion_id(&champ_name).ok_or("Not tracked")?;
            let ah = act.ability_haste.ok_or("Missing ability haste")?;
            state.engine.set_ability_haste(&id, ah)?;
            Ok(format!("{} ability haste {}", champ_name, ah))
        }
        "set_ability_rank" => {
            let id = state.engine.find_champion_id(&champ_name).ok_or("Not tracked")?;
            let ability = act.ability.ok_or("Missing ability")?;
            let rank = act.rank.ok_or("Missing rank")?;
            state.engine.set_ability_rank(&id, &ability, rank)?;
            Ok(format!("{} {} rank {}", champ_name, ability, rank))
        }
        "query_champion" => {
            let q = act
                .question
                .or(act.champion.map(|c| format!("Tell me about {}", c)))
                .unwrap_or_else(|| "cooldown rules".to_string());
            rag_query_inner(state, q).await
        }
        other => Err(format!("Unknown action: {}", other)),
    }
}

async fn rag_query_inner(state: &AppState, question: String) -> Result<String, String> {
    let chunks = if let Ok(emb) = openai::embed_query(&question).await {
        state
            .db
            .rag_search_semantic(&emb, 5)
            .unwrap_or_default()
    } else {
        vec![]
    };
    let chunks = if chunks.is_empty() {
        state.db.rag_search(&question, 5).map_err(|e| e.to_string())?
    } else {
        chunks
    };
    if chunks.is_empty() {
        return Ok("No champion data found. Run npm run ingest.".to_string());
    }
    let context = chunks
        .iter()
        .map(|c: &RagChunk| c.content.as_str())
        .collect::<Vec<_>>()
        .join("\n---\n");
    openai::answer_rag(&question, &context).await
}

#[tauri::command]
pub async fn rag_query(state: State<'_, AppState>, question: String) -> Result<String, String> {
    rag_query_inner(&state, question).await
}

#[tauri::command]
pub async fn speak_text(text: String) -> Result<String, String> {
    let audio = openai::text_to_speech(&text).await?;
    Ok(base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        audio,
    ))
}

#[tauri::command]
pub fn get_setting(state: State<'_, AppState>, key: String) -> Result<Option<String>, String> {
    let conn = state.db.conn.lock().map_err(|e| e.to_string())?;
    let val: Option<String> = conn
        .query_row(
            "SELECT value FROM settings WHERE key = ?1",
            rusqlite::params![key],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| e.to_string())?;
    Ok(val)
}

#[tauri::command]
pub fn set_setting(state: State<'_, AppState>, key: String, value: String) -> Result<(), String> {
    let conn = state.db.conn.lock().map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?1, ?2)",
        rusqlite::params![key, value],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
