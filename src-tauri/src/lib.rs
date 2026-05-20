mod commands;
mod cooldown;
mod db;
mod openai;

use commands::AppState;
use cooldown::{spawn_ticker, CooldownEngine};
use db::DbState;
use std::sync::Arc;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = dotenvy::dotenv();
    if let Ok(cwd) = std::env::current_dir() {
        let _ = dotenvy::from_path(cwd.join(".env"));
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(|app| {
            let db_path = DbState::resolve_db_path();
            let db = Arc::new(
                DbState::new(db_path.clone()).expect("Failed to open database"),
            );
            let engine = Arc::new(CooldownEngine::new(db.clone()));
            engine
                .load_persisted()
                .expect("Failed to load session");

            let engine_clone = engine.clone();
            let handle = app.handle().clone();
            spawn_ticker(engine_clone, handle);

            app.manage(AppState { db, engine });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_app_info,
            commands::list_champions,
            commands::get_active_champions,
            commands::add_champion,
            commands::remove_champion,
            commands::ability_used,
            commands::ability_used_by_name,
            commands::reset_ability,
            commands::set_level,
            commands::set_ability_haste,
            commands::set_ability_rank,
            commands::transcribe_voice,
            commands::process_voice,
            commands::rag_query,
            commands::speak_text,
            commands::get_setting,
            commands::set_setting,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
