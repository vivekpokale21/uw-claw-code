use std::fs;

use semantic_search::chunker::chunk_file;
use semantic_search::config::ChunkerConfig;
use tempfile::tempdir;

fn config() -> ChunkerConfig {
    ChunkerConfig {
        target_tokens: 128,
        overlap_tokens: 16,
        extensions: vec![".go".to_string(), ".py".to_string(), ".md".to_string()],
    }
}

#[test]
fn chunks_go_symbols_with_headers() {
    let dir = tempdir().expect("tempdir");
    let repo = dir.path();
    let go_file = repo.join("main.go");
    fs::write(
        &go_file,
        r#"package main

type Service struct {}

func Hello() string {
    return "hi"
}

func (s Service) Ping() bool {
    return true
}
"#,
    )
    .expect("write go file");

    let chunks = chunk_file(&go_file, repo, &config()).expect("chunk file");
    assert!(chunks.len() >= 3, "expected one chunk per major symbol");

    let symbols = chunks
        .iter()
        .map(|c| c.symbol_name.as_str())
        .collect::<Vec<_>>();
    assert!(symbols.contains(&"Service"));
    assert!(symbols.contains(&"Hello"));
    assert!(symbols.contains(&"Ping"));

    for chunk in &chunks {
        assert!(chunk.start_line > 0);
        assert!(chunk.end_line >= chunk.start_line);
        assert!(chunk.text_with_header.starts_with("# File:"));
        assert_eq!(chunk.id.len(), 64);
        assert!(chunk.id.chars().all(|c| c.is_ascii_hexdigit()));
    }
}

#[test]
fn chunks_python_class_and_methods() {
    let dir = tempdir().expect("tempdir");
    let repo = dir.path();
    let py_file = repo.join("main.py");
    fs::write(
        &py_file,
        r#"class AuthService:
    def login(self, user):
        return user

    def logout(self, user):
        return None
"#,
    )
    .expect("write py file");

    let chunks = chunk_file(&py_file, repo, &config()).expect("chunk file");
    let symbols = chunks
        .iter()
        .map(|c| c.symbol_name.as_str())
        .collect::<Vec<_>>();

    assert!(symbols.contains(&"AuthService"));
    assert!(symbols.contains(&"login"));
    assert!(symbols.contains(&"logout"));
}
