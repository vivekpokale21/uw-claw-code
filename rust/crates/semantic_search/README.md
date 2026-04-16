# semantic_search

Async semantic retrieval library for agentic code search in Claw.

## What It Provides

- AST-aware chunking (`go`, `python`, `typescript`) with fallback chunking for docs/config files.
- Incremental indexing with mtime tracking.
- Hybrid retrieval:
  - Dense vectors (`memory` mode local store, `http` mode Qdrant).
  - Sparse BM25 (native implementation).
  - RRF fusion.
- Optional HyDE query expansion and reranking via llama.cpp endpoints.
- Context assembly with dedupe + token budget truncation.

## Runtime Dependencies

- llama.cpp HTTP server exposing:
  - `/embedding`
  - `/completion` (HyDE)
  - `/reranking` (optional)
- Qdrant only when `qdrant.mode = "http"`.

Default local config is at [`config.toml`](./config.toml).

## Example Usage

```rust
use std::path::Path;
use semantic_search::{SearchPipeline, SearchRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut pipeline = SearchPipeline::new(Path::new("rust/crates/semantic_search/config.toml")).await?;
    pipeline.index(Path::new("/home/vivek/projects/dubai_boom_monitor")).await?;

    let result = pipeline
        .search(SearchRequest {
            query: "find where auth is handled".to_string(),
            ..Default::default()
        })
        .await?;

    println!("{}", result.context);
    Ok(())
}
```

## Mode Notes (`qdrant.mode`)

- `memory`:
  - Dense vectors are kept in local persisted state (`.semantic_search/dense_state.bin`).
  - No external Qdrant dependency.
- `http`:
  - Dense vectors/chunk payloads are upserted into Qdrant.
  - Dense retrieval is served by Qdrant query API with payload filters.
  - File deletion uses payload-filtered delete on `file_path`.

## Filter Mapping

`SearchFilters` maps to dense retrieval filters as:

- `path_prefix` -> `path_prefixes` payload match.
- `extensions` -> `file_ext` OR-match.
- `language` -> exact payload match.
- `modified_after` -> numeric range on `last_modified`.

## Integration Status With Existing Tool

Current `rust/crates/tools/src/semantic_search.rs` is still the legacy implementation.
To migrate tool execution to this crate:

1. Add `semantic_search` as a dependency in `rust/crates/tools/Cargo.toml`.
2. Replace `execute_semantic_search` internals with:
   - pipeline initialization (cached),
   - `index(...)` on startup and change events,
   - `search(...)` for requests.
3. Keep existing tool schema stable while mapping output fields from `SearchResult`.

## Verification

```bash
cd rust
cargo test -p semantic_search
```

