use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::UNIX_EPOCH;

use sha2::{Digest, Sha256};
use tokenizers::models::bpe::BPE;
use tokenizers::Tokenizer;
use tree_sitter::{Node, Parser};
use walkdir::WalkDir;

use crate::config::ChunkerConfig;
use crate::error::{Result, SearchError};
use crate::models::Chunk;

pub fn chunk_file(path: &Path, repo_root: &Path, config: &ChunkerConfig) -> Result<Vec<Chunk>> {
    let source = fs::read_to_string(path)?;
    let relative = relative_path(repo_root, path);
    let last_modified = file_modified_secs(path)?;
    let language = language_for_path(path);

    let mut chunks = if is_code_language(&language) {
        ast_chunks(path, &relative, &source, &language, last_modified)?
    } else {
        Vec::new()
    };

    if chunks.is_empty() {
        chunks = fallback_chunks(&relative, &source, &language, last_modified, config);
    }

    Ok(chunks)
}

pub fn chunk_repo(root: &Path, config: &ChunkerConfig) -> Result<Vec<Chunk>> {
    let allowed = config
        .extensions
        .iter()
        .map(|e| normalize_extension(e))
        .collect::<HashSet<_>>();

    let mut all = Vec::new();
    for entry in WalkDir::new(root).into_iter().filter_map(std::result::Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        if should_skip_dir(entry.path()) {
            continue;
        }
        if !matches_extension(entry.path(), &allowed) {
            continue;
        }
        let file_chunks = chunk_file(entry.path(), root, config)?;
        all.extend(file_chunks);
    }

    Ok(all)
}

fn should_skip_dir(path: &Path) -> bool {
    path.components().any(|part| {
        let s = part.as_os_str().to_string_lossy();
        matches!(
            s.as_ref(),
            ".git" | "target" | "node_modules" | "dist" | "build" | ".claw" | ".port_sessions"
        )
    })
}

fn normalize_extension(ext: &str) -> String {
    let trimmed = ext.trim().to_ascii_lowercase();
    if trimmed.starts_with('.') {
        trimmed
    } else {
        format!(".{trimmed}")
    }
}

fn matches_extension(path: &Path, allowed: &HashSet<String>) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e.to_ascii_lowercase()));
    ext.is_some_and(|e| allowed.contains(&e))
}

fn file_modified_secs(path: &Path) -> Result<u64> {
    let meta = fs::metadata(path)?;
    let modified = meta
        .modified()
        .map_err(|err| SearchError::ChunkError {
            path: path.display().to_string(),
            reason: format!("metadata.modified failed: {err}"),
        })?;
    let secs = modified
        .duration_since(UNIX_EPOCH)
        .map_err(|err| SearchError::ChunkError {
            path: path.display().to_string(),
            reason: format!("invalid modified time: {err}"),
        })?
        .as_secs();
    Ok(secs)
}

fn relative_path(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn language_for_path(path: &Path) -> String {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "go" => "go".to_string(),
        "py" => "python".to_string(),
        "ts" | "tsx" | "js" => "typescript".to_string(),
        "md" => "markdown".to_string(),
        "yaml" | "yml" => "yaml".to_string(),
        "toml" => "toml".to_string(),
        other => other.to_string(),
    }
}

fn is_code_language(language: &str) -> bool {
    matches!(language, "go" | "python" | "typescript")
}

fn ast_chunks(
    path: &Path,
    file_path: &str,
    source: &str,
    language: &str,
    last_modified: u64,
) -> Result<Vec<Chunk>> {
    let mut parser = Parser::new();
    let ts_language = match language {
        "go" => tree_sitter_go::language(),
        "python" => tree_sitter_python::language(),
        "typescript" => tree_sitter_typescript::language_typescript(),
        _ => {
            return Err(SearchError::ChunkError {
                path: path.display().to_string(),
                reason: format!("unsupported language: {language}"),
            })
        }
    };
    parser
        .set_language(&ts_language)
        .map_err(|err| SearchError::ChunkError {
            path: path.display().to_string(),
            reason: format!("parser set_language failed: {err}"),
        })?;

    let tree = parser.parse(source, None).ok_or_else(|| SearchError::ChunkError {
        path: path.display().to_string(),
        reason: "tree-sitter parse returned None".to_string(),
    })?;

    let source_bytes = source.as_bytes();
    let root = tree.root_node();
    let mut out = Vec::new();
    let mut stack = vec![root];

    while let Some(node) = stack.pop() {
        for i in (0..node.named_child_count()).rev() {
            if let Some(child) = node.named_child(i) {
                stack.push(child);
            }
        }

        if !is_symbol_node(language, node.kind()) {
            continue;
        }

        let symbol_name = extract_symbol_name(language, node, source_bytes)
            .unwrap_or_else(|| format!("symbol_{}", node.start_position().row + 1));
        let parent_name = extract_parent_name(language, node, source_bytes)
            .or_else(|| {
                Path::new(file_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(ToString::to_string)
            })
            .unwrap_or_else(|| "module".to_string());

        let start_line = node.start_position().row as usize + 1;
        let end_line = node.end_position().row as usize + 1;
        let text = source
            .get(node.byte_range())
            .unwrap_or_default()
            .to_string();
        if text.trim().is_empty() {
            continue;
        }

        out.push(build_chunk(
            file_path,
            language,
            &symbol_name,
            &parent_name,
            start_line,
            end_line,
            last_modified,
            text,
        ));
    }

    Ok(out)
}

fn is_symbol_node(language: &str, kind: &str) -> bool {
    match language {
        "go" => matches!(
            kind,
            "function_declaration" | "method_declaration" | "type_declaration"
        ),
        "python" => matches!(kind, "function_definition" | "class_definition"),
        "typescript" => matches!(
            kind,
            "function_declaration"
                | "method_definition"
                | "class_declaration"
                | "interface_declaration"
        ),
        _ => false,
    }
}

fn extract_symbol_name(language: &str, node: Node<'_>, source: &[u8]) -> Option<String> {
    if language == "go" && node.kind() == "type_declaration" {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if child.kind() == "type_spec" {
                if let Some(name) = child.child_by_field_name("name") {
                    return text_for_node(name, source);
                }
            }
        }
    }

    node.child_by_field_name("name")
        .and_then(|n| text_for_node(n, source))
}

fn extract_parent_name(language: &str, node: Node<'_>, source: &[u8]) -> Option<String> {
    let mut current = node.parent();
    while let Some(parent) = current {
        let kind = parent.kind();
        let is_parent = match language {
            "python" => kind == "class_definition",
            "typescript" => kind == "class_declaration",
            "go" => kind == "type_declaration",
            _ => false,
        };
        if is_parent {
            return parent
                .child_by_field_name("name")
                .and_then(|n| text_for_node(n, source));
        }
        current = parent.parent();
    }
    None
}

fn text_for_node(node: Node<'_>, source: &[u8]) -> Option<String> {
    std::str::from_utf8(&source[node.byte_range()])
        .ok()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
}

fn fallback_chunks(
    file_path: &str,
    source: &str,
    language: &str,
    last_modified: u64,
    config: &ChunkerConfig,
) -> Vec<Chunk> {
    let lines = source.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return Vec::new();
    }

    let target = config.target_tokens.max(32);
    let overlap = config.overlap_tokens.min(target.saturating_sub(1));

    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < lines.len() {
        let mut end = start;
        let mut token_count = 0usize;
        while end < lines.len() {
            let next = estimate_tokens(lines[end]);
            if token_count > 0 && token_count + next > target {
                break;
            }
            token_count += next;
            end += 1;
        }

        if end <= start {
            end = (start + 1).min(lines.len());
        }

        let text = lines[start..end].join("\n");
        let heading = nearest_heading(&lines, start)
            .unwrap_or_else(|| format!("L{}-L{}", start + 1, end));

        chunks.push(build_chunk(
            file_path,
            language,
            &heading,
            &module_name(file_path),
            start + 1,
            end,
            last_modified,
            text,
        ));

        if end == lines.len() {
            break;
        }

        let mut overlap_tokens_remaining = overlap;
        let mut new_start = end;
        while new_start > start {
            let prev_line = lines[new_start - 1];
            let prev_tokens = estimate_tokens(prev_line);
            if prev_tokens > overlap_tokens_remaining {
                break;
            }
            overlap_tokens_remaining = overlap_tokens_remaining.saturating_sub(prev_tokens);
            new_start -= 1;
        }

        if new_start == start {
            start = end;
        } else {
            start = new_start;
        }
    }

    chunks
}

fn nearest_heading(lines: &[&str], start: usize) -> Option<String> {
    for idx in (0..=start).rev() {
        let line = lines[idx].trim();
        if line.starts_with('#') {
            return Some(line.trim_start_matches('#').trim().to_string());
        }
    }
    None
}

fn module_name(file_path: &str) -> String {
    PathBuf::from(file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("module")
        .to_string()
}

fn build_chunk(
    file_path: &str,
    language: &str,
    symbol_name: &str,
    parent_name: &str,
    start_line: usize,
    end_line: usize,
    last_modified: u64,
    text: String,
) -> Chunk {
    let id = hash_chunk_id(file_path, start_line);
    let header = format!(
        "# File: {file_path} | Symbol: {parent_name}.{symbol_name} | Lines: {start_line}-{end_line}\n{text}"
    );
    Chunk {
        id,
        file_path: file_path.to_string(),
        language: language.to_string(),
        symbol_name: symbol_name.to_string(),
        parent_name: parent_name.to_string(),
        start_line,
        end_line,
        last_modified,
        text,
        text_with_header: header,
    }
}

fn hash_chunk_id(file_path: &str, start_line: usize) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{file_path}:{start_line}"));
    hex::encode(hasher.finalize())
}

fn estimate_tokens(text: &str) -> usize {
    static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();
    let tokenizer = TOKENIZER.get_or_init(|| Tokenizer::new(BPE::default()));
    tokenizer
        .encode(text, false)
        .map(|enc| enc.len())
        .unwrap_or_else(|_| text.split_whitespace().count().max(1))
}
