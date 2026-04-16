use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{Result, SearchError};

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub llamacpp: LlamacppConfig,
    pub qdrant: QdrantConfig,
    pub bm25: BM25Config,
    pub chunker: ChunkerConfig,
    pub retrieval: RetrievalConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlamacppConfig {
    pub base_url: String,
    pub embed_endpoint: String,
    pub completion_endpoint: String,
    pub rerank_endpoint: String,
    pub embed_model: String,
    pub hyde_model: String,
    pub rerank_model: String,
    pub request_timeout_secs: u64,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_vector_size")]
    pub vector_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QdrantConfig {
    pub mode: String,
    pub url: String,
    pub collection: String,
    pub persist_path: PathBuf,
    pub vector_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BM25Config {
    pub persist_path: PathBuf,
    pub k1: f32,
    pub b: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChunkerConfig {
    pub target_tokens: usize,
    pub overlap_tokens: usize,
    pub extensions: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RetrievalConfig {
    pub top_k: usize,
    pub rrf_k: usize,
    pub use_hyde: bool,
    pub use_rerank: bool,
    pub top_n: usize,
    pub max_tokens: usize,
}

fn default_batch_size() -> usize {
    32
}

fn default_vector_size() -> usize {
    768
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)?;
        let mut config: Config = toml::from_str(&raw)
            .map_err(|err| SearchError::ConfigError(format!("failed to parse config TOML: {err}")))?;
        config.validate()?;
        config.normalize();
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.llamacpp.base_url.trim().is_empty() {
            return Err(SearchError::ConfigError("llamacpp.base_url must not be empty".to_string()));
        }
        if self.llamacpp.request_timeout_secs == 0 {
            return Err(SearchError::ConfigError(
                "llamacpp.request_timeout_secs must be > 0".to_string(),
            ));
        }
        if self.retrieval.top_k == 0 || self.retrieval.top_n == 0 {
            return Err(SearchError::ConfigError(
                "retrieval.top_k and retrieval.top_n must be > 0".to_string(),
            ));
        }
        if self.chunker.target_tokens == 0 {
            return Err(SearchError::ConfigError(
                "chunker.target_tokens must be > 0".to_string(),
            ));
        }
        if self.chunker.extensions.is_empty() {
            return Err(SearchError::ConfigError(
                "chunker.extensions must not be empty".to_string(),
            ));
        }
        if self.qdrant.mode != "memory" && self.qdrant.mode != "http" {
            return Err(SearchError::ConfigError(
                "qdrant.mode must be either 'memory' or 'http'".to_string(),
            ));
        }
        if self.qdrant.mode == "http" && self.qdrant.url.trim().is_empty() {
            return Err(SearchError::ConfigError(
                "qdrant.url must not be empty when qdrant.mode='http'".to_string(),
            ));
        }
        Ok(())
    }

    fn normalize(&mut self) {
        self.llamacpp.base_url = self.llamacpp.base_url.trim_end_matches('/').to_string();
        self.llamacpp.embed_endpoint = ensure_leading_slash(&self.llamacpp.embed_endpoint);
        self.llamacpp.completion_endpoint = ensure_leading_slash(&self.llamacpp.completion_endpoint);
        self.llamacpp.rerank_endpoint = ensure_leading_slash(&self.llamacpp.rerank_endpoint);

        self.chunker.extensions = self
            .chunker
            .extensions
            .iter()
            .map(|ext| normalize_extension(ext))
            .collect();
        self.qdrant.mode = self.qdrant.mode.trim().to_ascii_lowercase();
        self.qdrant.url = self.qdrant.url.trim_end_matches('/').to_string();
    }
}

fn ensure_leading_slash(input: &str) -> String {
    if input.starts_with('/') {
        input.to_string()
    } else {
        format!("/{input}")
    }
}

fn normalize_extension(ext: &str) -> String {
    let trimmed = ext.trim();
    if trimmed.starts_with('.') {
        trimmed.to_ascii_lowercase()
    } else {
        format!(".{}", trimmed.to_ascii_lowercase())
    }
}
