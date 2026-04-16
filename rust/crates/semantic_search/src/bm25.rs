use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Result, SearchError};
use crate::models::Chunk;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DocumentStats {
    len: usize,
    term_freq: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Index {
    docs: HashMap<String, DocumentStats>,
    doc_freq: HashMap<String, usize>,
    avg_doc_len: f32,
    total_docs: usize,
    k1: f32,
    b: f32,
}

impl BM25Index {
    pub fn new() -> Self {
        Self {
            docs: HashMap::new(),
            doc_freq: HashMap::new(),
            avg_doc_len: 0.0,
            total_docs: 0,
            k1: 1.5,
            b: 0.75,
        }
    }

    pub fn with_params(k1: f32, b: f32) -> Self {
        let mut out = Self::new();
        out.k1 = k1;
        out.b = b;
        out
    }

    pub fn add_chunks(&mut self, chunks: &[Chunk]) {
        for chunk in chunks {
            self.add_chunk(chunk);
        }
    }

    pub fn remove_chunk(&mut self, id: &str) {
        if let Some(existing) = self.docs.remove(id) {
            for term in existing.term_freq.keys() {
                if let Some(df) = self.doc_freq.get_mut(term) {
                    *df = df.saturating_sub(1);
                    if *df == 0 {
                        self.doc_freq.remove(term);
                    }
                }
            }
            self.total_docs = self.docs.len();
            self.recompute_avg_doc_len();
        }
    }

    pub fn query(&self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        if top_k == 0 || self.docs.is_empty() {
            return Vec::new();
        }

        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let mut scores = Vec::with_capacity(self.docs.len());
        for (doc_id, doc) in &self.docs {
            let mut score = 0.0f32;
            for term in &query_terms {
                let tf = *doc.term_freq.get(term).unwrap_or(&0) as f32;
                if tf <= 0.0 {
                    continue;
                }

                let df = *self.doc_freq.get(term).unwrap_or(&0) as f32;
                if df <= 0.0 {
                    continue;
                }

                let idf = (((self.total_docs as f32 - df + 0.5) / (df + 0.5)) + 1.0).ln();
                let dl = doc.len as f32;
                let avgdl = self.avg_doc_len.max(1e-6);
                let denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl));
                score += idf * ((tf * (self.k1 + 1.0)) / denom.max(1e-6));
            }
            if score > 0.0 {
                scores.push((doc_id.clone(), score));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let bytes = bincode::serialize(self)
            .map_err(|err| SearchError::ConfigError(format!("BM25 serialize failed: {err}")))?;
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let index = bincode::deserialize(&bytes)
            .map_err(|err| SearchError::ConfigError(format!("BM25 deserialize failed: {err}")))?;
        Ok(index)
    }

    fn add_chunk(&mut self, chunk: &Chunk) {
        self.remove_chunk(&chunk.id);

        let tokens = tokenize(&chunk.text);
        let mut term_freq = HashMap::<String, usize>::new();
        for token in &tokens {
            *term_freq.entry(token.clone()).or_insert(0) += 1;
        }

        let unique_terms = term_freq.keys().cloned().collect::<Vec<_>>();
        for term in unique_terms {
            *self.doc_freq.entry(term).or_insert(0) += 1;
        }

        self.docs.insert(
            chunk.id.clone(),
            DocumentStats {
                len: tokens.len(),
                term_freq,
            },
        );
        self.total_docs = self.docs.len();
        self.recompute_avg_doc_len();
    }

    fn recompute_avg_doc_len(&mut self) {
        if self.docs.is_empty() {
            self.avg_doc_len = 0.0;
            return;
        }

        let total_len: usize = self.docs.values().map(|doc| doc.len).sum();
        self.avg_doc_len = total_len as f32 / self.docs.len() as f32;
    }
}

impl Default for BM25Index {
    fn default() -> Self {
        Self::new()
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect()
}
