#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Chunk {
    pub id: String,
    pub file_path: String,
    pub language: String,
    pub symbol_name: String,
    pub parent_name: String,
    pub start_line: usize,
    pub end_line: usize,
    pub last_modified: u64,
    pub text: String,
    pub text_with_header: String,
}

#[derive(Debug, Clone)]
pub struct RankedChunk {
    pub chunk: Chunk,
    pub score: f32,
    pub rank: usize,
}

#[derive(Debug, Default, Clone)]
pub struct SearchFilters {
    pub path_prefix: Option<String>,
    pub extensions: Option<Vec<String>>,
    pub language: Option<String>,
    pub modified_after: Option<u64>,
}

#[derive(Debug)]
pub struct SearchResult {
    pub context: String,
    pub chunks: Vec<RankedChunk>,
}
