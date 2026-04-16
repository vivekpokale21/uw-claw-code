use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::config::Config;
use crate::embedder;
use crate::error::Result;
use crate::index::IndexManager;
use crate::models::{Chunk, RankedChunk, SearchFilters};

pub struct Retriever<'a> {
    index: &'a IndexManager,
    config: &'a Config,
}

impl<'a> Retriever<'a> {
    pub fn new(index: &'a IndexManager, config: &'a Config) -> Self {
        Self { index, config }
    }

    pub async fn retrieve(
        &self,
        query: &str,
        query_embedding: Option<Vec<f32>>,
        filters: &SearchFilters,
        top_k: usize,
    ) -> Result<Vec<RankedChunk>> {
        let top_k = top_k.max(1);
        let embedding = if let Some(vec) = query_embedding {
            vec
        } else {
            embedder::embed_one(query, &self.config.llamacpp).await?
        };

        let dense_ids = self
            .index
            .dense_ranked_ids(&embedding, filters, top_k)
            .await?;
        let sparse_ids = self.sparse_ranked_ids(query, filters, top_k);

        let fused = rrf_fuse(&dense_ids, &sparse_ids, self.config.retrieval.rrf_k, top_k);

        let mut output = Vec::new();
        for (rank, (id, score)) in fused.into_iter().take(top_k).enumerate() {
            if let Some(chunk) = self.index.chunk_by_id(&id) {
                output.push(RankedChunk {
                    chunk: chunk.clone(),
                    score,
                    rank: rank + 1,
                });
            }
        }

        Ok(output)
    }

    fn sparse_ranked_ids(&self, query: &str, filters: &SearchFilters, top_k: usize) -> Vec<String> {
        self.index
            .bm25()
            .query(query, top_k * 4)
            .into_iter()
            .filter_map(|(id, _)| {
                let chunk = self.index.chunk_by_id(&id)?;
                if matches_filters(chunk, filters) {
                    Some(id)
                } else {
                    None
                }
            })
            .take(top_k)
            .collect()
    }
}

fn matches_filters(chunk: &Chunk, filters: &SearchFilters) -> bool {
    if let Some(prefix) = &filters.path_prefix {
        if !chunk.file_path.starts_with(prefix) {
            return false;
        }
    }

    if let Some(exts) = &filters.extensions {
        if !exts.iter().any(|ext| chunk.file_path.ends_with(ext)) {
            return false;
        }
    }

    if let Some(language) = &filters.language {
        if chunk.language != *language {
            return false;
        }
    }

    if let Some(modified_after) = filters.modified_after {
        if chunk.last_modified < modified_after {
            return false;
        }
    }

    true
}

pub fn rrf_fuse(
    dense_ids: &[String],
    sparse_ids: &[String],
    rrf_k: usize,
    top_k: usize,
) -> Vec<(String, f32)> {
    let fallback_rank = top_k + 1;

    let dense_rank = dense_ids
        .iter()
        .enumerate()
        .map(|(idx, id)| (id.clone(), idx + 1))
        .collect::<HashMap<_, _>>();
    let sparse_rank = sparse_ids
        .iter()
        .enumerate()
        .map(|(idx, id)| (id.clone(), idx + 1))
        .collect::<HashMap<_, _>>();

    let mut ids = HashSet::<String>::new();
    ids.extend(dense_ids.iter().cloned());
    ids.extend(sparse_ids.iter().cloned());

    let mut out = ids
        .into_iter()
        .map(|id| {
            let dense_r = *dense_rank.get(&id).unwrap_or(&fallback_rank) as f32;
            let sparse_r = *sparse_rank.get(&id).unwrap_or(&fallback_rank) as f32;
            let k = rrf_k as f32;
            let score = (1.0 / (k + dense_r)) + (1.0 / (k + sparse_r));
            (id, score)
        })
        .collect::<Vec<_>>();

    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    out
}
