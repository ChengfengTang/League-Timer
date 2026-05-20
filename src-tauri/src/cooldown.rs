use crate::db::{effective_cooldown, DbState, SpellInfo};
use chrono::Utc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TimerStatus {
    Idle,
    Ticking,
    Ready,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbilityState {
    pub key: String,
    pub rank: u8,
    pub status: TimerStatus,
    pub remaining_secs: f64,
    pub effective_cd: f64,
    pub base_cd: f64,
    pub spell_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveChampion {
    pub id: String,
    pub champion_key: String,
    pub name: String,
    pub level: u8,
    pub ability_haste: i32,
    pub abilities: Vec<AbilityState>,
}

#[derive(Debug, Clone)]
struct TimerEntry {
    champion_id: String,
    champion_name: String,
    ability_key: String,
    ends_at_ms: i64,
    effective_cd: f64,
}

pub struct CooldownEngine {
    db: Arc<DbState>,
    champions: RwLock<HashMap<String, ActiveChampion>>,
    timers: RwLock<HashMap<String, TimerEntry>>,
}

impl CooldownEngine {
    pub fn new(db: Arc<DbState>) -> Self {
        Self {
            db,
            champions: RwLock::new(HashMap::new()),
            timers: RwLock::new(HashMap::new()),
        }
    }

    pub fn load_persisted(&self) -> Result<(), String> {
        let conn = self.db.conn.lock().map_err(|e| e.to_string())?;
        let mut champs = self.champions.write();
        let mut timers = self.timers.write();
        champs.clear();
        timers.clear();

        let mut champ_stmt = conn
            .prepare(
                "SELECT id, champion_key, name, level, ability_haste, q_rank, w_rank, e_rank, r_rank FROM active_champions",
            )
            .map_err(|e| e.to_string())?;

        let champ_rows = champ_stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                ))
            })
            .map_err(|e| e.to_string())?;

        for row in champ_rows.flatten() {
            let (id, key, name, level, ah, qr, wr, er, rr) = row;
            let mut abilities = Vec::new();
            for (ab, rank) in [
                ("Q", qr),
                ("W", wr),
                ("E", er),
                ("R", rr),
            ] {
                let spell = self
                    .db
                    .get_spell(&key, ab)
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| SpellInfo {
                        champion_key: key.clone(),
                        champion_name: name.clone(),
                        ability_key: ab.to_string(),
                        spell_name: ab.to_string(),
                        cooldowns: vec![0.0],
                        cd_trigger: "on_cast".to_string(),
                        cd_delay_secs: 0.0,
                        cd_note: None,
                    });
                abilities.push(AbilityState {
                    key: ab.to_string(),
                    rank: rank.clamp(1, 5) as u8,
                    status: TimerStatus::Idle,
                    remaining_secs: 0.0,
                    effective_cd: 0.0,
                    base_cd: 0.0,
                    spell_name: spell.spell_name,
                });
            }
            champs.insert(
                id.clone(),
                ActiveChampion {
                    id,
                    champion_key: key,
                    name,
                    level: level.clamp(1, 18) as u8,
                    ability_haste: ah as i32,
                    abilities,
                },
            );
        }

        let mut timer_stmt = conn
            .prepare(
                "SELECT champion_id, ability_key, ends_at_ms, effective_cd, base_cd FROM active_timers",
            )
            .map_err(|e| e.to_string())?;

        let timer_rows = timer_stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, f64>(3)?,
                    row.get::<_, f64>(4)?,
                ))
            })
            .map_err(|e| e.to_string())?;

        let now = Utc::now().timestamp_millis();
        for row in timer_rows.flatten() {
            let (champ_id, ability, ends_at, eff, base) = row;
            if let Some(champ) = champs.get_mut(&champ_id) {
                let remaining = ((ends_at - now) as f64 / 1000.0).max(0.0);
                if let Some(ab) = champ.abilities.iter_mut().find(|a| a.key == ability) {
                    ab.effective_cd = eff;
                    ab.base_cd = base;
                    ab.remaining_secs = remaining;
                    ab.status = if remaining > 0.05 {
                        TimerStatus::Ticking
                    } else {
                        TimerStatus::Ready
                    };
                }
            }
            if let Some(champ) = champs.get(&champ_id) {
                let timer_id = format!("{}:{}", champ_id, ability);
                timers.insert(
                    timer_id,
                    TimerEntry {
                        champion_id: champ_id,
                        champion_name: champ.name.clone(),
                        ability_key: ability,
                        ends_at_ms: ends_at,
                        effective_cd: eff,
                    },
                );
            }
        }
        Ok(())
    }

    fn persist_champion(&self, champ: &ActiveChampion) -> Result<(), String> {
        let conn = self.db.conn.lock().map_err(|e| e.to_string())?;
        let (qr, wr, er, rr) = (
            rank_for(champ, "Q"),
            rank_for(champ, "W"),
            rank_for(champ, "E"),
            rank_for(champ, "R"),
        );
        conn.execute(
            r#"INSERT OR REPLACE INTO active_champions
               (id, champion_key, name, level, ability_haste, q_rank, w_rank, e_rank, r_rank)
               VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)"#,
            rusqlite::params![
                champ.id,
                champ.champion_key,
                champ.name,
                champ.level,
                champ.ability_haste,
                qr,
                wr,
                er,
                rr
            ],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn persist_timer(
        &self,
        champion_id: &str,
        ability: &str,
        ends_at_ms: i64,
        effective_cd: f64,
        base_cd: f64,
    ) -> Result<(), String> {
        let conn = self.db.conn.lock().map_err(|e| e.to_string())?;
        let id = format!("{}:{}", champion_id, ability);
        conn.execute(
            r#"INSERT OR REPLACE INTO active_timers
               (id, champion_id, ability_key, ends_at_ms, effective_cd, base_cd)
               VALUES (?1,?2,?3,?4,?5,?6)"#,
            rusqlite::params![id, champion_id, ability, ends_at_ms, effective_cd, base_cd],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn clear_timer_db(&self, champion_id: &str, ability: &str) -> Result<(), String> {
        let conn = self.db.conn.lock().map_err(|e| e.to_string())?;
        let id = format!("{}:{}", champion_id, ability);
        conn.execute("DELETE FROM active_timers WHERE id = ?1", rusqlite::params![id])
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn snapshot(&self) -> Vec<ActiveChampion> {
        self.sync_remaining();
        self.champions.read().values().cloned().collect()
    }

    fn sync_remaining(&self) {
        let now = Utc::now().timestamp_millis();
        let mut champs = self.champions.write();
        let timers = self.timers.read();
        for champ in champs.values_mut() {
            for ab in &mut champ.abilities {
                let tid = format!("{}:{}", champ.id, ab.key);
                if let Some(t) = timers.get(&tid) {
                    let rem = ((t.ends_at_ms - now) as f64 / 1000.0).max(0.0);
                    ab.remaining_secs = rem;
                    ab.status = if rem > 0.05 {
                        TimerStatus::Ticking
                    } else if ab.status == TimerStatus::Ticking {
                        TimerStatus::Ready
                    } else {
                        ab.status.clone()
                    };
                }
            }
        }
    }

    pub fn add_champion(&self, name: &str) -> Result<ActiveChampion, String> {
        let brief = self
            .db
            .find_champion_key(name)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("Unknown champion: {}", name))?;

        {
            let champs = self.champions.read();
            if champs.values().any(|c| c.champion_key == brief.ddragon_key) {
                return Err(format!("{} is already tracked", brief.name));
            }
        }

        let id = Uuid::new_v4().to_string();
        let mut abilities = Vec::new();
        for ab in ["Q", "W", "E", "R"] {
            let spell = self
                .db
                .get_spell(&brief.ddragon_key, ab)
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("Missing spell {} {}", brief.name, ab))?;
            abilities.push(AbilityState {
                key: ab.to_string(),
                rank: 1,
                status: TimerStatus::Idle,
                remaining_secs: 0.0,
                effective_cd: 0.0,
                base_cd: 0.0,
                spell_name: spell.spell_name,
            });
        }

        let champ = ActiveChampion {
            id: id.clone(),
            champion_key: brief.ddragon_key,
            name: brief.name,
            level: 1,
            ability_haste: 0,
            abilities,
        };
        self.persist_champion(&champ)?;
        self.champions.write().insert(id, champ.clone());
        Ok(champ)
    }

    pub fn remove_champion(&self, champion_id: &str) -> Result<(), String> {
        self.champions.write().remove(champion_id);
        self.timers
            .write()
            .retain(|_, t| t.champion_id != champion_id);
        let conn = self.db.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "DELETE FROM active_timers WHERE champion_id = ?1",
            rusqlite::params![champion_id],
        )
        .map_err(|e| e.to_string())?;
        conn.execute(
            "DELETE FROM active_champions WHERE id = ?1",
            rusqlite::params![champion_id],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn find_champion_id(&self, name: &str) -> Option<String> {
        let lower = name.to_lowercase();
        self.champions
            .read()
            .values()
            .find(|c| c.name.to_lowercase() == lower || c.id == name)
            .map(|c| c.id.clone())
    }

    pub fn set_level(&self, champion_id: &str, level: u8) -> Result<(), String> {
        let mut champs = self.champions.write();
        let champ = champs
            .get_mut(champion_id)
            .ok_or_else(|| "Champion not found".to_string())?;
        champ.level = level.clamp(1, 18);
        self.persist_champion(champ)?;
        Ok(())
    }

    pub fn set_ability_haste(&self, champion_id: &str, ah: i32) -> Result<(), String> {
        let mut champs = self.champions.write();
        let champ = champs
            .get_mut(champion_id)
            .ok_or_else(|| "Champion not found".to_string())?;
        champ.ability_haste = ah.max(0);
        self.persist_champion(champ)?;
        Ok(())
    }

    pub fn set_ability_rank(
        &self,
        champion_id: &str,
        ability: &str,
        rank: u8,
    ) -> Result<(), String> {
        let mut champs = self.champions.write();
        let champ = champs
            .get_mut(champion_id)
            .ok_or_else(|| "Champion not found".to_string())?;
        let ab = ability.to_uppercase();
        if let Some(state) = champ.abilities.iter_mut().find(|a| a.key == ab) {
            state.rank = rank.clamp(1, 5);
            self.persist_champion(champ)?;
            Ok(())
        } else {
            Err(format!("Unknown ability {}", ability))
        }
    }

    pub fn ability_used(
        &self,
        champion_id: &str,
        ability: &str,
        confirm_hit: bool,
    ) -> Result<(ActiveChampion, f64, String), String> {
        let ability = ability.to_uppercase();
        let (champ_key, champ_name, ah, rank) = {
            let champs = self.champions.read();
            let champ = champs
                .get(champion_id)
                .ok_or_else(|| "Champion not found".to_string())?;
            let rank = rank_for(champ, &ability);
            (
                champ.champion_key.clone(),
                champ.name.clone(),
                champ.ability_haste,
                rank,
            )
        };

        let spell = self
            .db
            .get_spell(&champ_key, &ability)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("Spell not found: {} {}", champ_name, ability))?;

        if spell.cd_trigger == "on_hit" && !confirm_hit {
            return Err(format!(
                "{} {} cooldown starts on hit — say \"{} {} landed\" to start timer",
                champ_name, ability, champ_name, ability
            ));
        }

        let idx = (rank as usize).saturating_sub(1);
        let base_cd = *spell
            .cooldowns
            .get(idx)
            .or_else(|| spell.cooldowns.last())
            .unwrap_or(&0.0);

        let delay_ms = (spell.cd_delay_secs * 1000.0) as i64;
        let effective = effective_cooldown(base_cd, ah);
        let duration_ms = (effective * 1000.0) as i64 + delay_ms;
        let ends_at = Utc::now().timestamp_millis() + duration_ms;

        {
            let mut champs = self.champions.write();
            if let Some(champ) = champs.get_mut(champion_id) {
                if let Some(ab) = champ.abilities.iter_mut().find(|a| a.key == ability) {
                    ab.status = if delay_ms > 0 {
                        TimerStatus::Idle
                    } else {
                        TimerStatus::Ticking
                    };
                    ab.remaining_secs = effective + spell.cd_delay_secs;
                    ab.effective_cd = effective;
                    ab.base_cd = base_cd;
                    ab.spell_name = spell.spell_name.clone();
                }
            }
        }

        let timer_id = format!("{}:{}", champion_id, ability);
        self.timers.write().insert(
            timer_id,
            TimerEntry {
                champion_id: champion_id.to_string(),
                champion_name: champ_name.clone(),
                ability_key: ability.clone(),
                ends_at_ms: ends_at,
                effective_cd: effective,
            },
        );
        self.persist_timer(champion_id, &ability, ends_at, effective, base_cd)?;

        let champ = self
            .champions
            .read()
            .get(champion_id)
            .cloned()
            .ok_or_else(|| "Champion not found".to_string())?;
        Ok((champ, effective, spell.cd_trigger))
    }

    pub fn reset_ability(&self, champion_id: &str, ability: &str) -> Result<(), String> {
        let ability = ability.to_uppercase();
        let timer_id = format!("{}:{}", champion_id, ability);
        self.timers.write().remove(&timer_id);
        self.clear_timer_db(champion_id, &ability)?;
        let mut champs = self.champions.write();
        if let Some(champ) = champs.get_mut(champion_id) {
            if let Some(ab) = champ.abilities.iter_mut().find(|a| a.key == ability) {
                ab.status = TimerStatus::Idle;
                ab.remaining_secs = 0.0;
            }
        }
        Ok(())
    }

    pub fn tick(&self, app: &AppHandle) -> Vec<(String, String)> {
        let now = Utc::now().timestamp_millis();
        let mut ready = Vec::new();
        let mut finished_ids = Vec::new();

        {
            let timers = self.timers.read();
            for (tid, t) in timers.iter() {
                if t.ends_at_ms <= now {
                    ready.push((t.champion_name.clone(), t.ability_key.clone()));
                    finished_ids.push(tid.clone());
                }
            }
        }

        if !finished_ids.is_empty() {
            let mut timers = self.timers.write();
            let mut champs = self.champions.write();
            for tid in &finished_ids {
                if let Some(t) = timers.remove(tid) {
                    let _ = self.clear_timer_db(&t.champion_id, &t.ability_key);
                    if let Some(champ) = champs.get_mut(&t.champion_id) {
                        if let Some(ab) = champ
                            .abilities
                            .iter_mut()
                            .find(|a| a.key == t.ability_key)
                        {
                            ab.status = TimerStatus::Ready;
                            ab.remaining_secs = 0.0;
                        }
                    }
                }
            }
        }

        let snapshot: Vec<ActiveChampion> = self.champions.read().values().cloned().collect();
        let _ = app.emit("cooldown-tick", snapshot);

        ready
    }
}

fn rank_for(champ: &ActiveChampion, ability: &str) -> i64 {
    champ
        .abilities
        .iter()
        .find(|a| a.key == ability)
        .map(|a| a.rank as i64)
        .unwrap_or(1)
}

pub fn spawn_ticker(engine: Arc<CooldownEngine>, app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
            let ready = engine.tick(&app);
            for (name, ability) in ready {
                let _ = app.emit(
                    "ability-ready",
                    serde_json::json!({
                        "champion": name,
                        "ability": ability,
                        "message": format!("{} {} back up", name, ability)
                    }),
                );
            }
        }
    });
}
