use std::cmp::Ordering;
use std::time::Duration;

use serde::Deserialize;
use serde_json::json;
use tracing::warn;

use crate::config::LlamacppConfig;
use crate::models::RankedChunk;

#[derive(Debug, Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

#[derive(Debug, Deserialize)]
struct RerankResult {
    index: usize,
    relevance_score: f32,
}

pub async fn rerank(
    query: &str,
    chunks: Vec<RankedChunk>,
    top_n: usize,
    config: &LlamacppConfig,
) -> Vec<RankedChunk> {
    if chunks.is_empty() {
        return chunks;
    }

    let endpoint = format!("{}{}", config.base_url, config.rerank_endpoint);
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(config.request_timeout_secs))
        .build()
    {
        Ok(client) => client,
        Err(err) => {
            warn!("reranker client init failed: {err}");
            return chunks.into_iter().take(top_n).collect();
        }
    };

    let docs = chunks
        .iter()
        .map(|chunk| chunk.chunk.text_with_header.clone())
        .collect::<Vec<_>>();

    let response = client
        .post(endpoint)
        .json(&json!({
            "model": config.rerank_model,
            "query": query,
            "documents": docs,
        }))
        .send()
        .await;

    let Ok(response) = response else {
        warn!("reranker request failed; returning non-reranked top_n");
        return chunks.into_iter().take(top_n).collect();
    };

    if !response.status().is_success() {
        warn!("reranker endpoint returned status {}; returning non-reranked top_n", response.status());
        return chunks.into_iter().take(top_n).collect();
    }

    let parsed = response.json::<RerankResponse>().await;
    let Ok(parsed) = parsed else {
        warn!("reranker response parse failed; returning non-reranked top_n");
        return chunks.into_iter().take(top_n).collect();
    };

    let mut scored = parsed
        .results
        .into_iter()
        .filter_map(|entry| chunks.get(entry.index).cloned().map(|chunk| (chunk, entry.relevance_score)))
        .collect::<Vec<_>>();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored
        .into_iter()
        .take(top_n)
        .enumerate()
        .map(|(idx, (mut chunk, score))| {
            chunk.rank = idx + 1;
            chunk.score = score;
            chunk
        })
        .collect()
}
