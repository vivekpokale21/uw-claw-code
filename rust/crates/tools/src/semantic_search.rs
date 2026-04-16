use std::fs;
use std::path::{Path, PathBuf};

use semantic_search_lib::config::{BM25Config, ChunkerConfig, Config, LlamacppConfig, QdrantConfig, RetrievalConfig};
use semantic_search_lib::{SearchFilters, SearchPipeline, SearchRequest};
use serde::{Deserialize, Serialize};

const DEFAULT_MAX_RESULTS: usize = 8;
const DEFAULT_CHUNK_LINES: usize = 80;
const DEFAULT_CHUNK_OVERLAP_LINES: usize = 20;

#[derive(Debug, Deserialize)]
pub struct SemanticSearchInput {
    pub query: String,
    pub path: Option<String>,
    pub max_results: Option<usize>,
    pub max_file_size_bytes: Option<usize>,
    pub chunk_lines: Option<usize>,
    pub chunk_overlap_lines: Option<usize>,
    pub extensions: Option<Vec<String>>,
    pub reindex: Option<bool>,
    pub use_embeddings: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct SemanticSearchOutput {
    pub query: String,
    pub root: String,
    pub mode: String,
    #[serde(rename = "indexRebuilt")]
    pub index_rebuilt: bool,
    #[serde(rename = "indexedFiles")]
    pub indexed_files: usize,
    #[serde(rename = "indexedChunks")]
    pub indexed_chunks: usize,
    #[serde(rename = "embeddingModel", skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    pub results: Vec<SemanticSearchResult>,
}

#[derive(Debug, Serialize)]
pub struct SemanticSearchResult {
    pub path: String,
    #[serde(rename = "startLine")]
    pub start_line: usize,
    #[serde(rename = "endLine")]
    pub end_line: usize,
    pub score: f64,
    pub snippet: String,
}

#[derive(Debug)]
struct QueryOptions {
    max_results: usize,
    chunk_lines: usize,
    chunk_overlap_lines: usize,
    extensions: Vec<String>,
    reindex: bool,
    use_embeddings: bool,
}

pub fn execute_semantic_search(input: SemanticSearchInput) -> Result<SemanticSearchOutput, String> {
    let query = input.query.trim().to_string();
    if query.is_empty() {
        return Err(String::from("query must not be empty"));
    }

    let root = resolve_root(input.path.as_deref())?;
    let options = parse_options(&input);
    let mut warnings = Vec::new();
    if input.max_file_size_bytes.is_some() {
        warnings.push(String::from(
            "max_file_size_bytes is not applied in the new semantic_search pipeline",
        ));
    }

    let cache_dir = root.join(".semantic_search");
    let had_cache = cache_dir.join("bm25.bin").exists() || cache_dir.join("dense_state.bin").exists();
    if options.reindex && cache_dir.exists() {
        let _ = fs::remove_dir_all(&cache_dir);
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| format!("semantic search runtime init failed: {err}"))?;

    let mut used_embeddings = options.use_embeddings;
    let first_try = runtime.block_on(run_pipeline_search(
        &root,
        &query,
        &options,
        options.use_embeddings,
    ));
    let (search_result, indexed_files, indexed_chunks, config) = match first_try {
        Ok(ok) => ok,
        Err(err) if options.use_embeddings && is_embedding_error(&err) => {
            warnings.push(format!(
                "embedding path failed ({err}); falling back to lexical semantic search"
            ));
            used_embeddings = false;
            runtime.block_on(run_pipeline_search(&root, &query, &options, false))?
        }
        Err(err) => return Err(err),
    };

    let results = search_result
        .chunks
        .iter()
        .take(options.max_results)
        .map(|entry| SemanticSearchResult {
            path: entry.chunk.file_path.clone(),
            start_line: entry.chunk.start_line,
            end_line: entry.chunk.end_line,
            score: ((entry.score as f64) * 1000.0).round() / 1000.0,
            snippet: truncate_snippet(&entry.chunk.text, 500),
        })
        .collect::<Vec<_>>();

    Ok(SemanticSearchOutput {
        query,
        root: root.display().to_string(),
        mode: if used_embeddings {
            String::from("embeddings")
        } else {
            String::from("lexical")
        },
        index_rebuilt: options.reindex || !had_cache,
        indexed_files,
        indexed_chunks,
        embedding_model: if used_embeddings {
            Some(config.llamacpp.embed_model)
        } else {
            None
        },
        warnings,
        results,
    })
}

async fn run_pipeline_search(
    root: &Path,
    query: &str,
    options: &QueryOptions,
    use_embeddings: bool,
) -> Result<(semantic_search_lib::SearchResult, usize, usize, Config), String> {
    let config = build_config(root, options, use_embeddings);
    let use_hyde = env_bool("CLAW_SEMANTIC_USE_HYDE", false) && use_embeddings;
    let use_rerank = env_bool("CLAW_SEMANTIC_USE_RERANK", false);

    let mut pipeline = SearchPipeline::from_config(config.clone())
        .await
        .map_err(|err| err.to_string())?;
    pipeline.index(root).await.map_err(|err| err.to_string())?;
    let (indexed_files, indexed_chunks) = pipeline.stats();
    let search_result = pipeline
        .search(SearchRequest {
            query: query.to_string(),
            filters: SearchFilters {
                extensions: Some(options.extensions.clone()),
                ..Default::default()
            },
            use_hyde: Some(use_hyde),
            use_rerank: Some(use_rerank),
            use_dense: Some(use_embeddings),
            top_k: Some((options.max_results.saturating_mul(3)).max(options.max_results)),
            top_n: Some(options.max_results),
            max_tokens: Some(config.retrieval.max_tokens),
        })
        .await
        .map_err(|err| err.to_string())?;
    Ok((search_result, indexed_files, indexed_chunks, config))
}

fn resolve_root(path: Option<&str>) -> Result<PathBuf, String> {
    match path {
        Some(raw) if !raw.trim().is_empty() => {
            let candidate = PathBuf::from(raw);
            let resolved = if candidate.is_absolute() {
                candidate
            } else {
                std::env::current_dir()
                    .map_err(|error| error.to_string())?
                    .join(candidate)
            };
            fs::canonicalize(&resolved).map_err(|error| {
                format!(
                    "failed to resolve semantic search root '{}': {error}",
                    resolved.display()
                )
            })
        }
        _ => std::env::current_dir().map_err(|error| error.to_string()),
    }
}

fn parse_options(input: &SemanticSearchInput) -> QueryOptions {
    QueryOptions {
        max_results: input.max_results.unwrap_or(DEFAULT_MAX_RESULTS).max(1),
        chunk_lines: input.chunk_lines.unwrap_or(DEFAULT_CHUNK_LINES).max(10),
        chunk_overlap_lines: input
            .chunk_overlap_lines
            .unwrap_or(DEFAULT_CHUNK_OVERLAP_LINES),
        extensions: normalize_extensions(input.extensions.as_deref()),
        reindex: input.reindex.unwrap_or(false),
        use_embeddings: input.use_embeddings.unwrap_or(true),
    }
}

fn normalize_extensions(extensions: Option<&[String]>) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(values) = extensions {
        for value in values {
            let cleaned = value.trim().trim_start_matches('.').to_ascii_lowercase();
            if !cleaned.is_empty() {
                out.push(format!(".{cleaned}"));
            }
        }
    }

    if out.is_empty() {
        for ext in [
            ".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".toml", ".yaml", ".yml",
            ".md", ".go", ".java", ".kt", ".kts", ".cpp", ".c", ".h", ".hpp", ".cs", ".rb",
            ".php", ".swift", ".sql", ".sh", ".bash", ".zsh", ".ps1", ".ini", ".cfg",
        ] {
            out.push(ext.to_string());
        }
    }
    out.sort();
    out.dedup();
    out
}

fn build_config(root: &Path, options: &QueryOptions, use_embeddings: bool) -> Config {
    let mut embed_base_url = std::env::var("CLAW_SEMANTIC_EMBED_BASE_URL")
        .ok()
        .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
        .unwrap_or_else(|| String::from("http://127.0.0.1:8129"));
    embed_base_url = embed_base_url.trim_end_matches('/').to_string();
    let embed_endpoint = std::env::var("CLAW_SEMANTIC_EMBED_ENDPOINT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| {
            if embed_base_url.ends_with("/v1") {
                String::from("/embeddings")
            } else {
                String::from("/embedding")
            }
        });

    let mut embed_model = std::env::var("CLAW_SEMANTIC_EMBED_MODEL")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| String::from("nomic-embed-text-v1.5"));
    if !use_embeddings {
        embed_model.clear();
    }
    let vector_size = std::env::var("CLAW_SEMANTIC_VECTOR_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(768);
    let use_hyde = env_bool("CLAW_SEMANTIC_USE_HYDE", false) && use_embeddings;
    let use_rerank = env_bool("CLAW_SEMANTIC_USE_RERANK", false);
    let qdrant_mode = std::env::var("CLAW_SEMANTIC_QDRANT_MODE")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .unwrap_or_else(|| String::from("memory"));
    let qdrant_url = std::env::var("CLAW_SEMANTIC_QDRANT_URL")
        .ok()
        .unwrap_or_else(|| String::from("http://127.0.0.1:6333"));

    Config {
        llamacpp: LlamacppConfig {
            base_url: embed_base_url,
            embed_endpoint,
            completion_endpoint: std::env::var("CLAW_SEMANTIC_COMPLETION_ENDPOINT")
                .ok()
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| String::from("/completion")),
            rerank_endpoint: std::env::var("CLAW_SEMANTIC_RERANK_ENDPOINT")
                .ok()
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| String::from("/reranking")),
            embed_model,
            hyde_model: std::env::var("CLAW_SEMANTIC_HYDE_MODEL")
                .ok()
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| String::from("qwen2.5-coder")),
            rerank_model: std::env::var("CLAW_SEMANTIC_RERANK_MODEL")
                .ok()
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| String::from("bge-reranker-base")),
            request_timeout_secs: std::env::var("CLAW_SEMANTIC_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(30),
            batch_size: std::env::var("CLAW_SEMANTIC_BATCH_SIZE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(16),
            vector_size,
        },
        qdrant: QdrantConfig {
            mode: qdrant_mode,
            url: qdrant_url,
            collection: qdrant_collection_for_root(root),
            persist_path: PathBuf::from(".semantic_search/qdrant"),
            vector_size,
        },
        bm25: BM25Config {
            persist_path: PathBuf::from(".semantic_search/bm25.bin"),
            k1: 1.5,
            b: 0.75,
        },
        chunker: ChunkerConfig {
            target_tokens: options.chunk_lines.saturating_mul(8).max(64),
            overlap_tokens: options.chunk_overlap_lines.saturating_mul(8),
            extensions: options.extensions.clone(),
        },
        retrieval: RetrievalConfig {
            top_k: (options.max_results.saturating_mul(3)).max(options.max_results),
            rrf_k: 60,
            use_hyde,
            use_rerank,
            top_n: options.max_results,
            max_tokens: 4096,
        },
    }
}

fn is_embedding_error(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("llama.cpp request failed")
        || lower.contains("embedding request")
        || lower.contains("/embedding")
        || lower.contains("/embeddings")
}

fn qdrant_collection_for_root(root: &Path) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    root.display().to_string().hash(&mut hasher);
    format!("codebase_{:x}", hasher.finish())
}

fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(value) => matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => default,
    }
}

fn truncate_snippet(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = text.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}

#[cfg(test)]
mod tests {
    use super::{execute_semantic_search, SemanticSearchInput};
    use std::fs;
    use std::path::PathBuf;

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("semantic-search-adapter-{label}-{nanos}"))
    }

    #[test]
    fn bm25_search_returns_relevant_file() {
        let root = temp_dir("bm25");
        fs::create_dir_all(root.join("app")).expect("create app dir");
        fs::write(
            root.join("app").join("api.py"),
            "def get_booms():\n    # apply endpoint rate limiting\n    return {}\n",
        )
        .expect("write api file");
        fs::write(
            root.join("README.md"),
            "This project tracks booms in Dubai.\n",
        )
        .expect("write readme");

        let output = execute_semantic_search(SemanticSearchInput {
            query: String::from("rate limit endpoint"),
            path: Some(root.display().to_string()),
            max_results: Some(3),
            max_file_size_bytes: None,
            chunk_lines: Some(30),
            chunk_overlap_lines: Some(5),
            extensions: None,
            reindex: Some(true),
            use_embeddings: Some(false),
        })
        .expect("semantic search should succeed");

        assert!(
            !output.results.is_empty(),
            "expected at least one semantic match"
        );
        assert_eq!(output.results[0].path, "app/api.py");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn cache_reuse_sets_index_rebuilt_false() {
        let root = temp_dir("cache");
        fs::create_dir_all(&root).expect("create root");
        fs::write(root.join("lib.rs"), "pub fn hello() {}\n").expect("write file");

        let first = execute_semantic_search(SemanticSearchInput {
            query: String::from("hello"),
            path: Some(root.display().to_string()),
            max_results: None,
            max_file_size_bytes: None,
            chunk_lines: None,
            chunk_overlap_lines: None,
            extensions: Some(vec![String::from("rs")]),
            reindex: Some(false),
            use_embeddings: Some(false),
        })
        .expect("first semantic search should succeed");
        assert!(first.index_rebuilt);

        let second = execute_semantic_search(SemanticSearchInput {
            query: String::from("hello"),
            path: Some(root.display().to_string()),
            max_results: None,
            max_file_size_bytes: None,
            chunk_lines: None,
            chunk_overlap_lines: None,
            extensions: Some(vec![String::from("rs")]),
            reindex: Some(false),
            use_embeddings: Some(false),
        })
        .expect("second semantic search should succeed");
        assert!(!second.index_rebuilt);

        let _ = fs::remove_dir_all(root);
    }
}
