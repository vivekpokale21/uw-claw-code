use std::fs;

use mockito::Matcher;
use semantic_search::pipeline::{SearchPipeline, SearchRequest};
use tempfile::tempdir;

#[tokio::test]
async fn pipeline_returns_context_and_chunks() {
    let mut server = mockito::Server::new_async().await;

    let _embed_mock = server
        .mock("POST", "/embedding")
        .match_body(Matcher::Regex(".*".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"embedding":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}"#)
        .create();

    let _completion_mock = server
        .mock("POST", "/completion")
        .match_body(Matcher::Regex(".*".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"content":"def authenticate(user):\n    return user is not None"}"#)
        .create();

    let dir = tempdir().expect("tempdir");
    let repo = dir.path().join("repo");
    fs::create_dir_all(repo.join("src")).expect("create repo");
    fs::write(
        repo.join("src").join("auth.py"),
        "def authenticate(user):\n    return user is not None\n",
    )
    .expect("write source");

    let config_path = dir.path().join("config.toml");
    fs::write(
        &config_path,
        format!(
            r#"[llamacpp]
base_url = "{}"
embed_endpoint = "/embedding"
completion_endpoint = "/completion"
rerank_endpoint = "/reranking"
embed_model = "nomic-embed-text-v1.5"
hyde_model = "qwen2.5-coder"
rerank_model = "bge-reranker-base"
request_timeout_secs = 10
batch_size = 4
vector_size = 8

[qdrant]
mode = "memory"
url = "http://localhost:6333"
collection = "codebase"
persist_path = ".semantic_search/qdrant"
vector_size = 8

[bm25]
persist_path = ".semantic_search/bm25.bin"
k1 = 1.5
b = 0.75

[chunker]
target_tokens = 128
overlap_tokens = 16
extensions = [".py", ".md"]

[retrieval]
top_k = 10
rrf_k = 60
use_hyde = true
use_rerank = false
top_n = 5
max_tokens = 1024
"#,
            server.url()
        ),
    )
    .expect("write config");

    let mut pipeline = SearchPipeline::new(&config_path)
        .await
        .expect("create pipeline");
    pipeline.index(&repo).await.expect("index repo");

    let result = pipeline
        .search(SearchRequest {
            query: "handle authentication".to_string(),
            ..Default::default()
        })
        .await
        .expect("search result");

    assert!(!result.context.trim().is_empty());
    assert!(result.context.contains("--- [1/"));
    assert!(!result.chunks.is_empty());
    assert!(result.chunks.iter().all(|c| c.score > 0.0));
}
