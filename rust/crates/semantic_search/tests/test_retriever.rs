use semantic_search::bm25::BM25Index;
use semantic_search::models::Chunk;
use semantic_search::retriever::rrf_fuse;

fn mk_chunk(id: &str, text: &str) -> Chunk {
    Chunk {
        id: id.to_string(),
        file_path: format!("src/{id}.rs"),
        language: "rust".to_string(),
        symbol_name: id.to_string(),
        parent_name: "mod".to_string(),
        start_line: 1,
        end_line: 5,
        last_modified: 0,
        text: text.to_string(),
        text_with_header: text.to_string(),
    }
}

#[test]
fn bm25_returns_expected_top_match() {
    let mut bm25 = BM25Index::new();
    let mut chunks = Vec::new();
    for idx in 0..10 {
        let id = format!("doc{idx}");
        let text = if idx == 2 {
            "authentication token validation middleware"
        } else if idx == 7 {
            "token refresh authentication flow"
        } else {
            "misc logging and helpers"
        };
        chunks.push(mk_chunk(&id, text));
    }

    bm25.add_chunks(&chunks);
    let hits = bm25.query("validation middleware", 3);
    assert!(!hits.is_empty());
    assert_eq!(hits[0].0, "doc2");
}

#[test]
fn rrf_fusion_prioritizes_present_candidates() {
    let dense = vec!["A".to_string(), "C".to_string(), "D".to_string()];
    let sparse = vec!["B".to_string(), "C".to_string(), "E".to_string()];
    let fused = rrf_fuse(&dense, &sparse, 60, 3);
    assert!(!fused.is_empty());
    let top = &fused[0].0;
    assert!(top == "A" || top == "B" || top == "C");
}
