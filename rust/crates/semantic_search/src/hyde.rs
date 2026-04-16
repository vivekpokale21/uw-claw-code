use std::hash::{Hash, Hasher};
use std::time::Duration;

use serde::Deserialize;
use serde_json::json;
use tracing::warn;

use crate::config::LlamacppConfig;
use crate::embedder;

#[derive(Debug, Deserialize)]
struct CompletionChoice {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CompletionResponse {
    content: Option<String>,
    completion: Option<String>,
    choices: Option<Vec<CompletionChoice>>,
}

pub async fn expand_query(query: &str, config: &LlamacppConfig) -> Vec<f32> {
    let plain_embedding = match embedder::embed_one(query, config).await {
        Ok(vec) => vec,
        Err(err) => {
            warn!("HyDE fallback query embedding failed: {err}");
            hashed_fallback(query, config.vector_size)
        }
    };

    let prompt = format!(
        "You are a code search assistant. Given a task or question about a codebase,\nwrite a short, realistic code snippet or function (10-30 lines) that would\nplausibly answer or implement it. Output only the code, no explanation.\n\nTask: {query}"
    );

    let completion = generate_hypothetical(&prompt, config).await;
    let hypo = match completion {
        Ok(text) if !text.trim().is_empty() => text,
        Ok(_) => query.to_string(),
        Err(err) => {
            warn!("HyDE completion failed: {err}; using plain query embedding");
            query.to_string()
        }
    };

    let hypo_embedding = match embedder::embed_one(&hypo, config).await {
        Ok(vec) => vec,
        Err(err) => {
            warn!("HyDE hypothetical embedding failed: {err}; using plain query embedding");
            plain_embedding.clone()
        }
    };

    average_vectors(&plain_embedding, &hypo_embedding, config.vector_size)
}

async fn generate_hypothetical(prompt: &str, config: &LlamacppConfig) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.request_timeout_secs))
        .build()
        .map_err(|err| format!("client build failed: {err}"))?;

    let response = client
        .post(format!("{}{}", config.base_url, config.completion_endpoint))
        .json(&json!({
            "prompt": prompt,
            "n_predict": 256,
            "temperature": 0.2,
            "stop": ["```"],
            "model": config.hyde_model,
        }))
        .send()
        .await
        .map_err(|err| format!("request failed: {err}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("status={status}, body={body}"));
    }

    let parsed = response
        .json::<CompletionResponse>()
        .await
        .map_err(|err| format!("invalid completion JSON: {err}"))?;

    let text = parsed
        .content
        .or(parsed.completion)
        .or_else(|| {
            parsed
                .choices
                .and_then(|choices| choices.into_iter().next())
                .and_then(|choice| choice.text)
        })
        .unwrap_or_default();

    Ok(text)
}

fn average_vectors(left: &[f32], right: &[f32], fallback_size: usize) -> Vec<f32> {
    if left.is_empty() && right.is_empty() {
        return vec![0.0; fallback_size.max(1)];
    }
    if left.is_empty() {
        return right.to_vec();
    }
    if right.is_empty() {
        return left.to_vec();
    }

    let len = left.len().min(right.len());
    (0..len)
        .map(|idx| (left[idx] + right[idx]) / 2.0)
        .collect()
}

fn hashed_fallback(seed: &str, size: usize) -> Vec<f32> {
    let size = size.max(1);
    let mut out = Vec::with_capacity(size);
    for idx in 0..size {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut hasher);
        idx.hash(&mut hasher);
        let value = hasher.finish() as f32 / u64::MAX as f32;
        out.push((value * 2.0) - 1.0);
    }
    out
}
