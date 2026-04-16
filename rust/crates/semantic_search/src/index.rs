use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use qdrant_client::qdrant::{
    point_id, Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter,
    PointId, PointStruct, QueryPointsBuilder, Range, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use serde::{Deserialize, Serialize};
use serde_json::json;
use walkdir::WalkDir;

use crate::bm25::BM25Index;
use crate::chunker::chunk_file;
use crate::config::Config;
use crate::embedder;
use crate::error::Result;
use crate::models::{Chunk, SearchFilters};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DenseStore {
    chunks_by_id: HashMap<String, Chunk>,
    vectors_by_id: HashMap<String, Vec<f32>>,
    file_to_chunk_ids: HashMap<String, HashSet<String>>,
    point_to_chunk: HashMap<String, String>,
    mtimes: HashMap<String, u64>,
}

pub struct IndexManager {
    config: Config,
    bm25: BM25Index,
    dense: DenseStore,
    qdrant: Option<Qdrant>,
    force_full_reindex: bool,
    repo_root: Option<PathBuf>,
}

impl IndexManager {
    pub async fn new(config: &Config) -> Result<Self> {
        let (qdrant, force_full_reindex) = if config.qdrant.mode.eq_ignore_ascii_case("http") {
            let client = Qdrant::from_url(&config.qdrant.url).build()?;
            let exists = client.collection_exists(&config.qdrant.collection).await?;
            if !exists {
                client
                    .create_collection(
                        CreateCollectionBuilder::new(&config.qdrant.collection).vectors_config(
                            VectorParamsBuilder::new(
                                config.qdrant.vector_size as u64,
                                Distance::Cosine,
                            ),
                        ),
                    )
                    .await?;
            }
            (Some(client), !exists)
        } else {
            (None, false)
        };

        let bm25 = BM25Index::with_params(config.bm25.k1, config.bm25.b);
        Ok(Self {
            config: config.clone(),
            bm25,
            dense: DenseStore::default(),
            qdrant,
            force_full_reindex,
            repo_root: None,
        })
    }

    pub async fn index_repo(&mut self, root: &Path) -> Result<()> {
        self.repo_root = Some(root.to_path_buf());
        self.load_state(root)?;

        let allowed_ext = self
            .config
            .chunker
            .extensions
            .iter()
            .map(|ext| ext.to_ascii_lowercase())
            .collect::<HashSet<_>>();

        let mut current_files = HashSet::<String>::new();
        let mut changed_files = Vec::<PathBuf>::new();

        for entry in WalkDir::new(root).into_iter().filter_map(std::result::Result::ok) {
            if !entry.file_type().is_file() {
                continue;
            }
            if should_skip(entry.path()) {
                continue;
            }

            let rel = entry
                .path()
                .strip_prefix(root)
                .unwrap_or(entry.path())
                .to_string_lossy()
                .replace('\\', "/");
            let ext = extension_of(entry.path());
            if !allowed_ext.contains(&ext) {
                continue;
            }

            let mtime = modified_secs(entry.path())?;
            let is_changed = self.force_full_reindex
                || self.dense.mtimes.get(&rel).copied() != Some(mtime);
            current_files.insert(rel.clone());
            if is_changed {
                changed_files.push(entry.path().to_path_buf());
            }
        }

        let deleted_files = self
            .dense
            .mtimes
            .keys()
            .filter(|path| !current_files.contains(*path))
            .cloned()
            .collect::<Vec<_>>();

        for deleted in deleted_files {
            self.delete_file_by_rel(&deleted).await?;
        }

        for file in changed_files {
            self.index_file(&file, root).await?;
        }

        self.save_state(root)?;
        self.force_full_reindex = false;
        Ok(())
    }

    pub async fn index_file(&mut self, path: &Path, root: &Path) -> Result<()> {
        self.repo_root = Some(root.to_path_buf());

        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        self.delete_file_by_rel(&rel).await?;

        let chunks = chunk_file(path, root, &self.config.chunker)?;
        let dense_enabled = !self.config.llamacpp.embed_model.trim().is_empty();
        let vectors = if dense_enabled {
            let text_inputs = chunks
                .iter()
                .map(|chunk| chunk.text_with_header.clone())
                .collect::<Vec<_>>();
            embedder::embed(&text_inputs, &self.config.llamacpp).await?
        } else {
            vec![Vec::new(); chunks.len()]
        };
        let keep_vectors_locally = self.qdrant.is_none();

        if dense_enabled {
            if let Some(qdrant) = &self.qdrant {
            let mut points = Vec::with_capacity(chunks.len());
            for (chunk, vector) in chunks.iter().zip(vectors.iter()) {
                let payload = Payload::try_from(json!({
                    "chunk_id": chunk.id,
                    "id": chunk.id,
                    "file_path": chunk.file_path,
                    "file_ext": file_extension(&chunk.file_path),
                    "path_prefixes": file_path_prefixes(&chunk.file_path),
                    "language": chunk.language,
                    "symbol_name": chunk.symbol_name,
                    "parent_name": chunk.parent_name,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "last_modified": chunk.last_modified,
                    "text": chunk.text,
                    "text_with_header": chunk.text_with_header,
                }))?;

                points.push(PointStruct::new(chunk_point_uuid(&chunk.id), vector.clone(), payload));
            }

            if !points.is_empty() {
                qdrant
                    .upsert_points(
                        UpsertPointsBuilder::new(&self.config.qdrant.collection, points).wait(true),
                    )
                    .await?;
            }
            }
        }

        for (chunk, vector) in chunks.iter().cloned().zip(vectors.into_iter()) {
            self.dense
                .file_to_chunk_ids
                .entry(rel.clone())
                .or_default()
                .insert(chunk.id.clone());
            self.dense
                .point_to_chunk
                .insert(point_key_for_chunk_id(&chunk.id), chunk.id.clone());
            if keep_vectors_locally && dense_enabled {
                self.dense.vectors_by_id.insert(chunk.id.clone(), vector);
            }
            self.dense
                .chunks_by_id
                .insert(chunk.id.clone(), chunk);
        }

        self.bm25.add_chunks(&chunks);
        self.dense.mtimes.insert(rel, modified_secs(path)?);
        Ok(())
    }

    pub async fn delete_file(&mut self, path: &Path) -> Result<()> {
        let rel = if let Some(root) = &self.repo_root {
            path.strip_prefix(root)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/")
        } else {
            path.to_string_lossy().replace('\\', "/")
        };
        self.delete_file_by_rel(&rel).await?;
        if let Some(root) = &self.repo_root {
            self.save_state(root)?;
        }
        Ok(())
    }

    pub fn bm25(&self) -> &BM25Index {
        &self.bm25
    }

    pub(crate) fn chunk_by_id(&self, id: &str) -> Option<&Chunk> {
        self.dense.chunks_by_id.get(id)
    }

    pub(crate) async fn dense_ranked_ids(
        &self,
        query_embedding: &[f32],
        filters: &SearchFilters,
        top_k: usize,
    ) -> Result<Vec<String>> {
        if query_embedding.is_empty() {
            return Ok(Vec::new());
        }

        if let Some(qdrant) = &self.qdrant {
            let mut request = QueryPointsBuilder::new(&self.config.qdrant.collection)
                .query(query_embedding.to_vec())
                .limit(top_k as u64)
                .with_payload(false);

            if let Some(filter) = to_qdrant_filter(filters) {
                request = request.filter(filter);
            }

            let response = qdrant.query(request).await?;
            let mut ranked = Vec::with_capacity(response.result.len());
            for point in response.result {
                let Some(point_id) = point.id else {
                    continue;
                };
                let Some(key) = point_key_from_point_id(&point_id) else {
                    continue;
                };
                if let Some(chunk_id) = self.dense.point_to_chunk.get(&key) {
                    ranked.push(chunk_id.clone());
                }
            }
            Ok(ranked)
        } else {
            Ok(self.memory_dense_ranked_ids(query_embedding, filters, top_k))
        }
    }

    pub fn stats(&self) -> (usize, usize) {
        (self.dense.mtimes.len(), self.dense.chunks_by_id.len())
    }

    fn memory_dense_ranked_ids(
        &self,
        query_embedding: &[f32],
        filters: &SearchFilters,
        top_k: usize,
    ) -> Vec<String> {
        let mut scored = self
            .dense
            .chunks_by_id
            .values()
            .filter(|chunk| matches_filters(chunk, filters))
            .filter_map(|chunk| {
                let vector = self.dense.vectors_by_id.get(&chunk.id)?;
                Some((chunk.id.clone(), cosine_similarity(query_embedding, vector)))
            })
            .collect::<Vec<_>>();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.into_iter().take(top_k).map(|(id, _)| id).collect()
    }

    async fn delete_file_by_rel(&mut self, rel: &str) -> Result<()> {
        if let Some(ids) = self.dense.file_to_chunk_ids.remove(rel) {
            if let Some(qdrant) = &self.qdrant {
                let filter = Filter::all([Condition::matches("file_path", rel.to_string())]);
                qdrant
                    .delete_points(
                        DeletePointsBuilder::new(&self.config.qdrant.collection)
                            .points(filter)
                            .wait(true),
                    )
                    .await?;
            }

            for id in ids {
                self.dense.chunks_by_id.remove(&id);
                self.dense.vectors_by_id.remove(&id);
                self.dense.point_to_chunk.remove(&point_key_for_chunk_id(&id));
                self.bm25.remove_chunk(&id);
            }
        }
        self.dense.mtimes.remove(rel);
        Ok(())
    }

    fn load_state(&mut self, root: &Path) -> Result<()> {
        let bm25_path = root.join(&self.config.bm25.persist_path);
        let dense_path = dense_state_path(root);

        if bm25_path.exists() {
            self.bm25 = BM25Index::load(&bm25_path)?;
        }
        if dense_path.exists() {
            let bytes = fs::read(&dense_path)?;
            self.dense = bincode::deserialize(&bytes)
                .map_err(|err| crate::error::SearchError::ConfigError(format!("dense-state deserialize failed: {err}")))?;
        }

        Ok(())
    }

    fn save_state(&self, root: &Path) -> Result<()> {
        let bm25_path = root.join(&self.config.bm25.persist_path);
        self.bm25.save(&bm25_path)?;

        let dense_path = dense_state_path(root);
        if let Some(parent) = dense_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let bytes = bincode::serialize(&self.dense)
            .map_err(|err| crate::error::SearchError::ConfigError(format!("dense-state serialize failed: {err}")))?;
        fs::write(dense_path, bytes)?;
        Ok(())
    }
}

fn dense_state_path(root: &Path) -> PathBuf {
    root.join(".semantic_search").join("dense_state.bin")
}

fn should_skip(path: &Path) -> bool {
    path.components().any(|part| {
        let s = part.as_os_str().to_string_lossy();
        matches!(
            s.as_ref(),
            ".git" | "target" | "node_modules" | "dist" | "build" | ".claw" | ".port_sessions"
        )
    })
}

fn extension_of(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| format!(".{}", ext.to_ascii_lowercase()))
        .unwrap_or_default()
}

fn modified_secs(path: &Path) -> Result<u64> {
    let modified = fs::metadata(path)?.modified()?;
    let secs = modified
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|err| crate::error::SearchError::ConfigError(format!("invalid file mtime: {err}")))?
        .as_secs();
    Ok(secs)
}

fn file_extension(file_path: &str) -> String {
    Path::new(file_path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| format!(".{}", ext.to_ascii_lowercase()))
        .unwrap_or_default()
}

fn file_path_prefixes(file_path: &str) -> Vec<String> {
    let normalized = file_path.trim_matches('/');
    if normalized.is_empty() {
        return Vec::new();
    }

    let parts = normalized.split('/').collect::<Vec<_>>();
    let mut prefixes = Vec::with_capacity(parts.len());
    for idx in 0..parts.len() {
        prefixes.push(parts[..=idx].join("/"));
    }
    prefixes
}

fn to_qdrant_filter(filters: &SearchFilters) -> Option<Filter> {
    let mut must = Vec::<Condition>::new();

    if let Some(prefix) = &filters.path_prefix {
        let normalized = prefix.trim().trim_end_matches('/').to_string();
        if !normalized.is_empty() {
            must.push(Condition::matches("path_prefixes", normalized));
        }
    }

    if let Some(extensions) = &filters.extensions {
        let ext_conditions = extensions
            .iter()
            .map(|ext| normalize_extension(ext))
            .filter(|ext| !ext.is_empty())
            .map(|ext| Condition::matches("file_ext", ext))
            .collect::<Vec<_>>();
        if !ext_conditions.is_empty() {
            must.push(Filter::any(ext_conditions).into());
        }
    }

    if let Some(language) = &filters.language {
        let language = language.trim();
        if !language.is_empty() {
            must.push(Condition::matches("language", language.to_string()));
        }
    }

    if let Some(modified_after) = filters.modified_after {
        must.push(Condition::range(
            "last_modified",
            Range {
                gte: Some(modified_after as f64),
                ..Default::default()
            },
        ));
    }

    if must.is_empty() {
        None
    } else {
        Some(Filter::all(must))
    }
}

fn normalize_extension(ext: &str) -> String {
    let trimmed = ext.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.starts_with('.') {
        trimmed.to_ascii_lowercase()
    } else {
        format!(".{}", trimmed.to_ascii_lowercase())
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

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;

    for (a, b) in left.iter().zip(right.iter()) {
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }

    let denom = (left_norm.sqrt() * right_norm.sqrt()).max(1e-12);
    dot / denom
}

fn chunk_point_uuid(chunk_id: &str) -> String {
    let mut hex = chunk_id
        .chars()
        .filter(|ch| ch.is_ascii_hexdigit())
        .take(32)
        .collect::<String>();
    while hex.len() < 32 {
        hex.push('0');
    }

    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

fn point_key_for_chunk_id(chunk_id: &str) -> String {
    format!("uuid:{}", chunk_point_uuid(chunk_id))
}

fn point_key_from_point_id(point_id: &PointId) -> Option<String> {
    match &point_id.point_id_options {
        Some(point_id::PointIdOptions::Num(id)) => Some(format!("num:{id}")),
        Some(point_id::PointIdOptions::Uuid(id)) => Some(format!("uuid:{id}")),
        None => None,
    }
}
