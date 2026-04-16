use std::path::{Path, PathBuf};

use crate::assembler;
use crate::config::Config;
use crate::embedder;
use crate::error::Result;
use crate::hyde;
use crate::index::IndexManager;
use crate::models::{SearchFilters, SearchResult};
use crate::reranker;
use crate::retriever::Retriever;

pub struct SearchPipeline {
    config: Config,
    index: IndexManager,
    repo_root: Option<PathBuf>,
}

#[derive(Debug, Default)]
pub struct SearchRequest {
    pub query: String,
    pub filters: SearchFilters,
    pub use_hyde: Option<bool>,
    pub use_rerank: Option<bool>,
    pub use_dense: Option<bool>,
    pub top_k: Option<usize>,
    pub top_n: Option<usize>,
    pub max_tokens: Option<usize>,
}

impl SearchPipeline {
    pub async fn from_config(config: Config) -> Result<Self> {
        let index = IndexManager::new(&config).await?;
        Ok(Self {
            config,
            index,
            repo_root: None,
        })
    }

    pub async fn new(config_path: &Path) -> Result<Self> {
        let config = Config::load(config_path)?;
        Self::from_config(config).await
    }

    pub async fn index(&mut self, repo_root: &Path) -> Result<()> {
        self.repo_root = Some(repo_root.to_path_buf());
        self.index.index_repo(repo_root).await
    }

    pub async fn search(&self, request: SearchRequest) -> Result<SearchResult> {
        let query = request.query.trim();
        let use_hyde = request.use_hyde.unwrap_or(self.config.retrieval.use_hyde);
        let use_rerank = request.use_rerank.unwrap_or(self.config.retrieval.use_rerank);
        let use_dense = request.use_dense.unwrap_or(true);
        let top_k = request.top_k.unwrap_or(self.config.retrieval.top_k);
        let top_n = request.top_n.unwrap_or(self.config.retrieval.top_n);
        let max_tokens = request.max_tokens.unwrap_or(self.config.retrieval.max_tokens);

        let query_embedding = if !use_dense {
            Some(Vec::new())
        } else if use_hyde {
            Some(hyde::expand_query(query, &self.config.llamacpp).await)
        } else {
            Some(embedder::embed_one(query, &self.config.llamacpp).await?)
        };

        let retriever = Retriever::new(&self.index, &self.config);
        let retrieved = retriever
            .retrieve(query, query_embedding, &request.filters, top_k)
            .await?;

        let reranked = if use_rerank {
            reranker::rerank(query, retrieved, top_n, &self.config.llamacpp).await
        } else {
            retrieved.into_iter().take(top_n).collect()
        };

        let repo_root = self
            .repo_root
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let context = assembler::assemble(&reranked, max_tokens, &repo_root)?;

        Ok(SearchResult {
            context,
            chunks: reranked,
        })
    }

    pub fn stats(&self) -> (usize, usize) {
        self.index.stats()
    }
}
