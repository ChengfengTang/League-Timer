use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize)]
pub struct VoiceAction {
    pub action: String,
    pub champion: Option<String>,
    pub ability: Option<String>,
    pub level: Option<u8>,
    pub ability_haste: Option<i32>,
    pub rank: Option<u8>,
    pub question: Option<String>,
    pub confirm_hit: Option<bool>,
}

fn api_key() -> Result<String, String> {
    dotenvy::dotenv().ok();
    std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY not set".to_string())
}

pub async fn transcribe_audio(audio_base64: &str, mime: &str) -> Result<String, String> {
    let key = api_key()?;
    let bytes = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        audio_base64,
    )
    .map_err(|e| e.to_string())?;

    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name("audio.webm")
        .mime_str(mime)
        .map_err(|e| e.to_string())?;
    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", "whisper-1");

    let client = reqwest::Client::new();
    let res = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .bearer_auth(&key)
        .multipart(form)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let body: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;
    if let Some(err) = body.get("error") {
        return Err(err.to_string());
    }
    body["text"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "No transcript".to_string())
}

pub async fn parse_voice_command(
    transcript: &str,
    active_champions: &[String],
) -> Result<Vec<VoiceAction>, String> {
    let key = api_key()?;
    let roster = if active_champions.is_empty() {
        "none".to_string()
    } else {
        active_champions.join(", ")
    };

    let system = format!(
        r#"You parse League of Legends cooldown tracker voice commands into JSON actions.
Active champions: {roster}
Rules:
- NEVER invent cooldown numbers.
- Return a JSON object with key "actions" containing an array of actions.
- Actions: add_champion, remove_champion, ability_used, ability_hit (same as ability_used with confirm_hit), ability_ready_ack, set_level, set_ability_haste, set_ability_rank, query_champion.
- ability keys: Q, W, E, R, D, F.
- "ahri used E", "ahri E", "E down on ahri" -> ability_used.
- "ahri E landed" -> ability_hit.
- "add champion ahri" -> add_champion.
- "ahri level 11" -> set_level.
- "40 ability haste on ahri" -> set_ability_haste.
- "ahri E rank 3" -> set_ability_rank.
- Questions about cooldown rules -> query_champion."#
    );

    let client = reqwest::Client::new();
    let res = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(&key)
        .json(&json!({
            "model": "gpt-4o-mini",
            "temperature": 0,
            "response_format": { "type": "json_object" },
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": transcript }
            ]
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let body: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| body.to_string())?;

    let parsed: serde_json::Value =
        serde_json::from_str(content).map_err(|e| format!("Parse error: {} — {}", e, content))?;

    let arr = parsed
        .get("actions")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_else(|| parsed.as_array().cloned().unwrap_or_default());

    let mut actions = Vec::new();
    for item in arr {
        let action = item
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        actions.push(VoiceAction {
            action,
            champion: item.get("champion").and_then(|v| v.as_str()).map(String::from),
            ability: item.get("ability").and_then(|v| v.as_str()).map(String::from),
            level: item.get("level").and_then(|v| v.as_u64()).map(|n| n as u8),
            ability_haste: item
                .get("ability_haste")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32),
            rank: item.get("rank").and_then(|v| v.as_u64()).map(|n| n as u8),
            question: item.get("question").and_then(|v| v.as_str()).map(String::from),
            confirm_hit: item.get("confirm_hit").and_then(|v| v.as_bool()),
        });
    }
    Ok(actions)
}

pub async fn embed_query(text: &str) -> Result<Vec<f32>, String> {
    let key = api_key()?;
    let client = reqwest::Client::new();
    let res = client
        .post("https://api.openai.com/v1/embeddings")
        .bearer_auth(&key)
        .json(&json!({
            "model": "text-embedding-3-small",
            "input": text
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let body: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;
    let emb = body["data"][0]["embedding"]
        .as_array()
        .ok_or_else(|| "No embedding".to_string())?;
    Ok(emb
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect())
}

pub async fn answer_rag(question: &str, context: &str) -> Result<String, String> {
    let key = api_key()?;
    let client = reqwest::Client::new();
    let res = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(&key)
        .json(&json!({
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "messages": [
                { "role": "system", "content": "Answer using only the provided champion ability context. Be concise." },
                { "role": "user", "content": format!("Context:\n{context}\n\nQuestion: {question}") }
            ]
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let body: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;
    body["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "No answer".to_string())
}

pub async fn text_to_speech(text: &str) -> Result<Vec<u8>, String> {
    let key = api_key()?;
    let client = reqwest::Client::new();
    let res = client
        .post("https://api.openai.com/v1/audio/speech")
        .bearer_auth(&key)
        .json(&json!({
            "model": "tts-1",
            "voice": "nova",
            "input": text
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let bytes = res.bytes().await.map_err(|e| e.to_string())?;
    Ok(bytes.to_vec())
}
