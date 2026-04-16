#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    #[error("llama.cpp request failed: {0}")]
    LlamacppError(String),

    #[error("Qdrant error: {0}")]
    QdrantError(#[from] qdrant_client::QdrantError),

    #[error("Chunking failed for {path}: {reason}")]
    ChunkError { path: String, reason: String },

    #[error("Config error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, SearchError>;
