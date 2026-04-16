use std::sync::Arc;
use std::time::Duration;

use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tokio::time::sleep;

use crate::config::LlamacppConfig;
use crate::error::{Result, SearchError};

#[derive(Debug, Deserialize)]
struct LlamaEmbeddingResponse {
    embedding: Option<Vec<f32>>,
    data: Option<Vec<LlamaEmbeddingData>>,
}

#[derive(Debug, Deserialize)]
struct LlamaEmbeddingData {
    embedding: Vec<f32>,
}

pub async fn embed(texts: &[String], config: &LlamacppConfig) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let timeout = Duration::from_secs(config.request_timeout_secs);
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|err| SearchError::LlamacppError(format!("client build failed: {err}")))?;

    let concurrency = config.batch_size.max(1);
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut join_set = JoinSet::new();

    for (idx, text) in texts.iter().enumerate() {
        let permit = semaphore.clone().acquire_owned().await.map_err(|err| {
            SearchError::LlamacppError(format!("semaphore acquire failed: {err}"))
        })?;
        let client = client.clone();
        let endpoint = format!("{}{}", config.base_url, config.embed_endpoint);
        let text = text.clone();
        let model = config.embed_model.clone();
        join_set.spawn(async move {
            let _permit = permit;
            let result = embed_with_retry(&client, &endpoint, &text, &model).await;
            (idx, result)
        });
    }

    let mut output = vec![Vec::<f32>::new(); texts.len()];
    while let Some(join_result) = join_set.join_next().await {
        let (idx, result) = join_result
            .map_err(|err| SearchError::LlamacppError(format!("embed join failed: {err}")))?;
        output[idx] = result?;
    }

    Ok(output)
}

pub async fn embed_one(text: &str, config: &LlamacppConfig) -> Result<Vec<f32>> {
    let vectors = embed(&[text.to_string()], config).await?;
    vectors
        .into_iter()
        .next()
        .ok_or_else(|| SearchError::LlamacppError("embedding response was empty".to_string()))
}

async fn embed_with_retry(
    client: &reqwest::Client,
    endpoint: &str,
    text: &str,
    model: &str,
) -> Result<Vec<f32>> {
    let mut delay = Duration::from_millis(200);
    let payloads = embedding_payloads(text, model);

    for attempt in 0..=3 {
        let mut last_error: Option<SearchError> = None;

        for payload in &payloads {
            let response = client.post(endpoint).json(payload).send().await;

            match response {
                Ok(resp) => {
                    if resp.status().is_success() {
                        return parse_embedding_response(resp).await;
                    }

                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    last_error = Some(SearchError::LlamacppError(format!(
                        "embedding request failed ({status}): {body}"
                    )));
                    if is_retryable_status(status) {
                        continue;
                    }
                }
                Err(err) => {
                    last_error = Some(SearchError::LlamacppError(format!(
                        "embedding request transport failure: {err}"
                    )));
                }
            }
        }

        if attempt == 3 {
            return Err(last_error.unwrap_or_else(|| {
                SearchError::LlamacppError("embedding retries exhausted".to_string())
            }));
        }

        sleep(delay).await;
        delay = delay.saturating_mul(2);
    }

    Err(SearchError::LlamacppError(
        "embedding retries exhausted".to_string(),
    ))
}

fn is_retryable_status(status: StatusCode) -> bool {
    matches!(status.as_u16(), 500 | 502 | 503 | 504)
}

fn embedding_payloads(text: &str, model: &str) -> Vec<serde_json::Value> {
    let mut payloads = vec![json!({ "content": text })];
    if !model.trim().is_empty() {
        payloads.push(json!({ "model": model, "input": text }));
    } else {
        payloads.push(json!({ "input": text }));
    }
    payloads
}

async fn parse_embedding_response(resp: reqwest::Response) -> Result<Vec<f32>> {
    let parsed: LlamaEmbeddingResponse = resp
        .json()
        .await
        .map_err(|err| SearchError::LlamacppError(format!("invalid embedding JSON: {err}")))?;

    if let Some(embedding) = parsed.embedding {
        if !embedding.is_empty() {
            return Ok(embedding);
        }
    }

    if let Some(mut data) = parsed.data {
        if let Some(first) = data.drain(..).next() {
            if !first.embedding.is_empty() {
                return Ok(first.embedding);
            }
        }
    }

    Err(SearchError::LlamacppError(
        "embedding response contained no embedding vector".to_string(),
    ))
}
