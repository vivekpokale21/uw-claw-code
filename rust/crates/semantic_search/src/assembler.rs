use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use tokenizers::models::bpe::BPE;
use tokenizers::Tokenizer;

use crate::error::Result;
use crate::models::RankedChunk;

pub fn assemble(chunks: &[RankedChunk], max_tokens: usize, repo_root: &Path) -> Result<String> {
    let max_tokens = max_tokens.max(1);
    let deduped = dedupe_chunks(chunks);

    let mut selected = Vec::<(RankedChunk, String)>::new();
    let mut running_tokens = estimate_tokens("Found 0 relevant chunks across 0 files.");

    for chunk in deduped {
        let with_bleed = with_context_bleed(&chunk, repo_root);
        let block = format!(
            "--- [{{idx}}/{{total}}] File: {} | Symbol: {}.{} | Lines: {}-{} ---\n{}\n",
            chunk.chunk.file_path,
            chunk.chunk.parent_name,
            chunk.chunk.symbol_name,
            chunk.chunk.start_line,
            chunk.chunk.end_line,
            with_bleed
        );
        let block_tokens = estimate_tokens(&block);
        if running_tokens + block_tokens > max_tokens {
            break;
        }

        running_tokens += block_tokens;
        selected.push((chunk, block));
    }

    let mut files = HashSet::<String>::new();
    for (chunk, _) in &selected {
        files.insert(chunk.chunk.file_path.clone());
    }

    let mut out = String::new();
    out.push_str(&format!(
        "Found {} relevant chunks across {} files.\n\n",
        selected.len(),
        files.len()
    ));

    let total = selected.len();
    for (idx, (_, block)) in selected.into_iter().enumerate() {
        let rendered = block
            .replace("{idx}", &(idx + 1).to_string())
            .replace("{total}", &total.to_string());
        out.push_str(&rendered);
        out.push('\n');
    }

    Ok(out)
}

fn dedupe_chunks(chunks: &[RankedChunk]) -> Vec<RankedChunk> {
    let mut seen = HashSet::<(String, String)>::new();
    let mut out = Vec::new();

    for chunk in chunks {
        let key = (chunk.chunk.file_path.clone(), chunk.chunk.symbol_name.clone());
        if seen.insert(key) {
            out.push(chunk.clone());
        }
    }

    out
}

fn with_context_bleed(chunk: &RankedChunk, repo_root: &Path) -> String {
    let path = repo_root.join(&chunk.chunk.file_path);
    let Ok(raw) = fs::read_to_string(path) else {
        return chunk.chunk.text.clone();
    };
    let lines = raw.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return chunk.chunk.text.clone();
    }

    let start = chunk.chunk.start_line.saturating_sub(2).max(1);
    let end = (chunk.chunk.end_line + 2).min(lines.len());
    lines[start - 1..end].join("\n")
}

fn estimate_tokens(text: &str) -> usize {
    static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();
    let tokenizer = TOKENIZER.get_or_init(|| Tokenizer::new(BPE::default()));
    tokenizer
        .encode(text, false)
        .map(|enc| enc.len())
        .unwrap_or_else(|_| text.split_whitespace().count().max(1))
}
