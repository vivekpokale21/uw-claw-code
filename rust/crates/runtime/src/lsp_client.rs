#![allow(clippy::should_implement_trait, clippy::must_use_candidate)]
//! LSP (Language Server Protocol) client registry for tool dispatch.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Supported LSP actions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LspAction {
    Diagnostics,
    Health,
    Hover,
    Definition,
    References,
    Completion,
    Symbols,
    Format,
}

impl LspAction {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "diagnostics" => Some(Self::Diagnostics),
            "health" | "status" => Some(Self::Health),
            "hover" => Some(Self::Hover),
            "definition" | "goto_definition" => Some(Self::Definition),
            "references" | "find_references" => Some(Self::References),
            "completion" | "completions" => Some(Self::Completion),
            "symbols" | "document_symbols" => Some(Self::Symbols),
            "format" | "formatting" => Some(Self::Format),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspDiagnostic {
    pub path: String,
    pub line: u32,
    pub character: u32,
    pub severity: String,
    pub message: String,
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspLocation {
    pub path: String,
    pub line: u32,
    pub character: u32,
    pub end_line: Option<u32>,
    pub end_character: Option<u32>,
    pub preview: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspHoverResult {
    pub content: String,
    pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspCompletionItem {
    pub label: String,
    pub kind: Option<String>,
    pub detail: Option<String>,
    pub insert_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspSymbol {
    pub name: String,
    pub kind: String,
    pub path: String,
    pub line: u32,
    pub character: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LspServerStatus {
    Connected,
    Disconnected,
    Starting,
    Error,
}

impl std::fmt::Display for LspServerStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connected => write!(f, "connected"),
            Self::Disconnected => write!(f, "disconnected"),
            Self::Starting => write!(f, "starting"),
            Self::Error => write!(f, "error"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspServerState {
    pub language: String,
    pub status: LspServerStatus,
    pub root_path: Option<String>,
    pub capabilities: Vec<String>,
    pub diagnostics: Vec<LspDiagnostic>,
}

const DEFAULT_LSP_HEALTH_STATE_FILE: &str = "lsp_health_state.json";
const DEFAULT_LSP_STATE_DIR: &str = ".port_sessions";
const DEFAULT_LSP_HEALTH_VERSION: u32 = 1;
const DEFAULT_LSP_MAX_CONSECUTIVE_FAILURES: u32 = 3;
const DEFAULT_LSP_COOLDOWN_SECONDS: f64 = 300.0;
const ENV_LSP_HEALTH_STATE_FILE: &str = "LSP_HEALTH_STATE_FILE";
const ENV_CLAW_LSP_HEALTH_STATE_FILE: &str = "CLAW_LSP_HEALTH_STATE_FILE";
const ENV_LSP_MAX_CONSECUTIVE_FAILURES: &str = "LSP_MAX_CONSECUTIVE_FAILURES";
const ENV_LSP_COOLDOWN_SECONDS: &str = "LSP_COOLDOWN_SECONDS";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct LspHealthState {
    pub consecutive_failures: u32,
    pub blocked_until_unix: f64,
    pub last_error: String,
    pub last_attempt_unix: f64,
    pub last_success_unix: f64,
    pub total_attempts: u64,
    pub total_failures: u64,
    pub last_warning: String,
    pub last_capabilities: String,
    pub last_failure_kind: String,
    pub recent_crash_loops: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedLspHealthPayload {
    #[serde(default = "default_lsp_health_version")]
    version: u32,
    #[serde(default)]
    saved_unix: f64,
    #[serde(default)]
    health: HashMap<String, LspHealthState>,
}

const fn default_lsp_health_version() -> u32 {
    DEFAULT_LSP_HEALTH_VERSION
}

#[derive(Debug, Clone, Default)]
pub struct LspRegistry {
    inner: Arc<Mutex<RegistryInner>>,
}

#[derive(Debug, Default)]
struct RegistryInner {
    servers: HashMap<String, LspServerState>,
    health: HashMap<String, LspHealthState>,
}

impl LspRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &self,
        language: &str,
        status: LspServerStatus,
        root_path: Option<&str>,
        capabilities: Vec<String>,
    ) {
        let mut inner = self.inner.lock().expect("lsp registry lock poisoned");
        inner.servers.insert(
            language.to_owned(),
            LspServerState {
                language: language.to_owned(),
                status,
                root_path: root_path.map(str::to_owned),
                capabilities,
                diagnostics: Vec::new(),
            },
        );
    }

    pub fn get(&self, language: &str) -> Option<LspServerState> {
        let inner = self.inner.lock().expect("lsp registry lock poisoned");
        inner.servers.get(language).cloned()
    }

    /// Find the appropriate server for a file path based on extension.
    pub fn find_server_for_path(&self, path: &str) -> Option<LspServerState> {
        language_for_path(path).and_then(|language| self.get(language))
    }

    /// List all registered servers.
    pub fn list_servers(&self) -> Vec<LspServerState> {
        let inner = self.inner.lock().expect("lsp registry lock poisoned");
        inner.servers.values().cloned().collect()
    }

    /// Add diagnostics to a server.
    pub fn add_diagnostics(
        &self,
        language: &str,
        diagnostics: Vec<LspDiagnostic>,
    ) -> Result<(), String> {
        let mut inner = self.inner.lock().expect("lsp registry lock poisoned");
        let server = inner
            .servers
            .get_mut(language)
            .ok_or_else(|| format!("LSP server not found for language: {language}"))?;
        server.diagnostics.extend(diagnostics);
        Ok(())
    }

    /// Get diagnostics for a specific file path.
    pub fn get_diagnostics(&self, path: &str) -> Vec<LspDiagnostic> {
        let inner = self.inner.lock().expect("lsp registry lock poisoned");
        inner
            .servers
            .values()
            .flat_map(|s| &s.diagnostics)
            .filter(|d| d.path == path)
            .cloned()
            .collect()
    }

    /// Clear diagnostics for a language server.
    pub fn clear_diagnostics(&self, language: &str) -> Result<(), String> {
        let mut inner = self.inner.lock().expect("lsp registry lock poisoned");
        let server = inner
            .servers
            .get_mut(language)
            .ok_or_else(|| format!("LSP server not found for language: {language}"))?;
        server.diagnostics.clear();
        Ok(())
    }

    /// Disconnect a server.
    pub fn disconnect(&self, language: &str) -> Option<LspServerState> {
        let mut inner = self.inner.lock().expect("lsp registry lock poisoned");
        inner.servers.remove(language)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        let inner = self.inner.lock().expect("lsp registry lock poisoned");
        inner.servers.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot current LSP health state.
    pub fn health_snapshot(&self) -> HashMap<String, LspHealthState> {
        let inner = self.inner.lock().expect("lsp registry lock poisoned");
        inner.health.clone()
    }

    /// Load persisted health state from a JSON file.
    pub fn load_health_from_path(&self, path: &Path) -> Result<usize, String> {
        if !path.exists() {
            return Ok(0);
        }
        let raw = fs::read_to_string(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let payload: PersistedLspHealthPayload = serde_json::from_str(&raw)
            .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
        let loaded = payload.health.len();
        let mut inner = self.inner.lock().expect("lsp registry lock poisoned");
        for (key, state) in payload.health {
            inner.health.insert(key, state);
        }
        Ok(loaded)
    }

    /// Persist health state to a JSON file.
    pub fn persist_health_to_path(&self, path: &Path) -> Result<(), String> {
        let health = {
            let inner = self.inner.lock().expect("lsp registry lock poisoned");
            inner.health.clone()
        };
        let payload = PersistedLspHealthPayload {
            version: DEFAULT_LSP_HEALTH_VERSION,
            saved_unix: now_unix_seconds(),
            health,
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        }
        let serialized = serde_json::to_string_pretty(&payload)
            .map_err(|error| format!("failed to serialize health payload: {error}"))?;
        fs::write(path, serialized)
            .map_err(|error| format!("failed to write {}: {error}", path.display()))
    }

    #[must_use]
    pub fn default_health_state_path(repo_root: Option<&Path>) -> PathBuf {
        if let Some(raw) = std::env::var_os(ENV_CLAW_LSP_HEALTH_STATE_FILE)
            .or_else(|| std::env::var_os(ENV_LSP_HEALTH_STATE_FILE))
        {
            let candidate = PathBuf::from(raw);
            if candidate.is_absolute() {
                return candidate;
            }
            let root = repo_root
                .map(Path::to_path_buf)
                .or_else(|| std::env::current_dir().ok())
                .unwrap_or_else(|| PathBuf::from("."));
            return root.join(candidate);
        }

        let root = repo_root
            .map(Path::to_path_buf)
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."));
        root.join(DEFAULT_LSP_STATE_DIR)
            .join(DEFAULT_LSP_HEALTH_STATE_FILE)
    }

    pub fn load_health_from_default_path(&self, repo_root: Option<&Path>) -> Result<usize, String> {
        let path = Self::default_health_state_path(repo_root);
        self.load_health_from_path(&path)
    }

    pub fn persist_health_to_default_path(
        &self,
        repo_root: Option<&Path>,
    ) -> Result<PathBuf, String> {
        let path = Self::default_health_state_path(repo_root);
        self.persist_health_to_path(&path)?;
        Ok(path)
    }

    /// Dispatch an LSP action and return a structured result.
    pub fn dispatch(
        &self,
        action: &str,
        path: Option<&str>,
        line: Option<u32>,
        character: Option<u32>,
        _query: Option<&str>,
    ) -> Result<serde_json::Value, String> {
        let lsp_action =
            LspAction::from_str(action).ok_or_else(|| format!("unknown LSP action: {action}"))?;

        if lsp_action == LspAction::Health {
            return Ok(self.health_status_payload());
        }

        // For diagnostics, we can check existing cached diagnostics
        if lsp_action == LspAction::Diagnostics {
            if let Some(path) = path {
                let diags = self.get_diagnostics(path);
                return Ok(serde_json::json!({
                    "action": "diagnostics",
                    "path": path,
                    "diagnostics": diags,
                    "count": diags.len()
                }));
            }
            // All diagnostics across all servers
            let inner = self.inner.lock().expect("lsp registry lock poisoned");
            let all_diags: Vec<_> = inner
                .servers
                .values()
                .flat_map(|s| &s.diagnostics)
                .collect();
            return Ok(serde_json::json!({
                "action": "diagnostics",
                "diagnostics": all_diags,
                "count": all_diags.len()
            }));
        }

        // For other actions, we need a connected server for the given file
        let path = path.ok_or("path is required for this LSP action")?;
        let health_key = language_for_path(path).map(str::to_owned);

        if let Some(key) = health_key.as_deref() {
            if let Some(cooldown_remaining) = self.cooldown_remaining_seconds(key) {
                let message = self
                    .last_health_error(key)
                    .map_or_else(String::new, |error| format!(" after error: {error}"));
                return Err(format!(
                    "LSP action blocked for '{key}': cooldown active for {cooldown_remaining:.1}s{message}"
                ));
            }
            self.record_attempt(key);
        }

        let server = match self.find_server_for_path(path) {
            Some(server) => server,
            None => {
                if let Some(key) = health_key.as_deref() {
                    self.record_failure(
                        key,
                        "request",
                        format!("no LSP server available for path: {path}"),
                    );
                }
                return Err(format!("no LSP server available for path: {path}"));
            }
        };

        if server.status != LspServerStatus::Connected {
            if let Some(key) = health_key.as_deref() {
                self.record_failure(
                    key,
                    "startup",
                    format!(
                        "LSP server for '{}' is not connected (status: {})",
                        server.language, server.status
                    ),
                );
            }
            return Err(format!(
                "LSP server for '{}' is not connected (status: {})",
                server.language, server.status
            ));
        }

        if let Some(key) = health_key.as_deref() {
            self.record_success(key, &server.capabilities);
        }

        // Return structured placeholder — actual LSP JSON-RPC calls would
        // go through the real LSP process here.
        Ok(serde_json::json!({
            "action": action,
            "path": path,
            "line": line,
            "character": character,
            "language": server.language,
            "status": "dispatched",
            "message": format!("LSP {} dispatched to {} server", action, server.language)
        }))
    }

    fn record_attempt(&self, health_key: &str) {
        let mut inner = self.inner.lock().expect("lsp registry lock poisoned");
        let state = inner.health.entry(health_key.to_string()).or_default();
        state.total_attempts = state.total_attempts.saturating_add(1);
        state.last_attempt_unix = now_unix_seconds();
    }

    fn record_success(&self, health_key: &str, capabilities: &[String]) {
        let mut inner = self.inner.lock().expect("lsp registry lock poisoned");
        let state = inner.health.entry(health_key.to_string()).or_default();
        state.consecutive_failures = 0;
        state.blocked_until_unix = 0.0;
        state.last_success_unix = now_unix_seconds();
        state.last_failure_kind.clear();
        state.recent_crash_loops = 0;
        state.last_warning.clear();
        state.last_error.clear();
        state.last_capabilities = summarize_capabilities(capabilities);
    }

    fn record_failure(&self, health_key: &str, failure_kind: &str, error: String) {
        let now = now_unix_seconds();
        let max_failures = lsp_max_consecutive_failures();
        let cooldown = lsp_cooldown_seconds();
        let mut inner = self.inner.lock().expect("lsp registry lock poisoned");
        let state = inner.health.entry(health_key.to_string()).or_default();
        state.consecutive_failures = state.consecutive_failures.saturating_add(1);
        state.total_failures = state.total_failures.saturating_add(1);
        state.last_error = error;
        state.last_failure_kind = failure_kind.to_string();
        if state.consecutive_failures >= max_failures {
            state.blocked_until_unix = now + cooldown;
        }
    }

    fn cooldown_remaining_seconds(&self, health_key: &str) -> Option<f64> {
        let now = now_unix_seconds();
        let inner = self.inner.lock().expect("lsp registry lock poisoned");
        let state = inner.health.get(health_key)?;
        if state.blocked_until_unix <= now {
            return None;
        }
        Some(state.blocked_until_unix - now)
    }

    fn last_health_error(&self, health_key: &str) -> Option<String> {
        let inner = self.inner.lock().expect("lsp registry lock poisoned");
        inner
            .health
            .get(health_key)
            .map(|state| state.last_error.clone())
            .filter(|value| !value.is_empty())
    }

    fn health_status_payload(&self) -> serde_json::Value {
        let now = now_unix_seconds();
        let inner = self.inner.lock().expect("lsp registry lock poisoned");
        let mut health = serde_json::Map::new();
        let mut keys = inner.health.keys().cloned().collect::<Vec<_>>();
        keys.sort();
        for key in keys {
            if let Some(state) = inner.health.get(&key) {
                let cooldown_remaining = (state.blocked_until_unix - now).max(0.0);
                health.insert(
                    key,
                    serde_json::json!({
                        "consecutive_failures": state.consecutive_failures,
                        "cooldown_remaining_seconds": cooldown_remaining,
                        "last_error": state.last_error,
                        "last_attempt_unix": state.last_attempt_unix,
                        "last_success_unix": state.last_success_unix,
                        "total_attempts": state.total_attempts,
                        "total_failures": state.total_failures,
                        "last_warning": state.last_warning,
                        "last_capabilities": state.last_capabilities,
                        "last_failure_kind": state.last_failure_kind,
                        "recent_crash_loops": state.recent_crash_loops
                    }),
                );
            }
        }
        let mut servers = inner.servers.values().cloned().collect::<Vec<_>>();
        servers.sort_by(|left, right| left.language.cmp(&right.language));
        serde_json::json!({
            "action": "health",
            "servers": servers,
            "health": health,
            "count": health.len()
        })
    }
}

fn language_for_path(path: &str) -> Option<&'static str> {
    let ext = Path::new(path)
        .extension()
        .and_then(|value| value.to_str())?;
    match ext {
        "rs" => Some("rust"),
        "ts" | "tsx" => Some("typescript"),
        "js" | "jsx" => Some("javascript"),
        "py" => Some("python"),
        "go" => Some("go"),
        "java" => Some("java"),
        "c" | "h" => Some("c"),
        "cpp" | "hpp" | "cc" => Some("cpp"),
        "rb" => Some("ruby"),
        "lua" => Some("lua"),
        _ => None,
    }
}

fn summarize_capabilities(capabilities: &[String]) -> String {
    if capabilities.is_empty() {
        return "none".to_string();
    }
    let mut sorted = capabilities.to_vec();
    sorted.sort();
    sorted.dedup();
    sorted.join(",")
}

fn lsp_max_consecutive_failures() -> u32 {
    std::env::var(ENV_LSP_MAX_CONSECUTIVE_FAILURES)
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .map(|value| value.max(1))
        .unwrap_or(DEFAULT_LSP_MAX_CONSECUTIVE_FAILURES)
}

fn lsp_cooldown_seconds() -> f64 {
    std::env::var(ENV_LSP_COOLDOWN_SECONDS)
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .map(|value| value.max(1.0))
        .unwrap_or(DEFAULT_LSP_COOLDOWN_SECONDS)
}

fn now_unix_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_and_retrieves_server() {
        let registry = LspRegistry::new();
        registry.register(
            "rust",
            LspServerStatus::Connected,
            Some("/workspace"),
            vec!["hover".into(), "completion".into()],
        );

        let server = registry.get("rust").expect("should exist");
        assert_eq!(server.language, "rust");
        assert_eq!(server.status, LspServerStatus::Connected);
        assert_eq!(server.capabilities.len(), 2);
    }

    #[test]
    fn finds_server_by_file_extension() {
        let registry = LspRegistry::new();
        registry.register("rust", LspServerStatus::Connected, None, vec![]);
        registry.register("typescript", LspServerStatus::Connected, None, vec![]);

        let rs_server = registry.find_server_for_path("src/main.rs").unwrap();
        assert_eq!(rs_server.language, "rust");

        let ts_server = registry.find_server_for_path("src/index.ts").unwrap();
        assert_eq!(ts_server.language, "typescript");

        assert!(registry.find_server_for_path("data.csv").is_none());
    }

    #[test]
    fn manages_diagnostics() {
        let registry = LspRegistry::new();
        registry.register("rust", LspServerStatus::Connected, None, vec![]);

        registry
            .add_diagnostics(
                "rust",
                vec![LspDiagnostic {
                    path: "src/main.rs".into(),
                    line: 10,
                    character: 5,
                    severity: "error".into(),
                    message: "mismatched types".into(),
                    source: Some("rust-analyzer".into()),
                }],
            )
            .unwrap();

        let diags = registry.get_diagnostics("src/main.rs");
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].message, "mismatched types");

        registry.clear_diagnostics("rust").unwrap();
        assert!(registry.get_diagnostics("src/main.rs").is_empty());
    }

    #[test]
    fn dispatches_diagnostics_action() {
        let registry = LspRegistry::new();
        registry.register("rust", LspServerStatus::Connected, None, vec![]);
        registry
            .add_diagnostics(
                "rust",
                vec![LspDiagnostic {
                    path: "src/lib.rs".into(),
                    line: 1,
                    character: 0,
                    severity: "warning".into(),
                    message: "unused import".into(),
                    source: None,
                }],
            )
            .unwrap();

        let result = registry
            .dispatch("diagnostics", Some("src/lib.rs"), None, None, None)
            .unwrap();
        assert_eq!(result["count"], 1);
    }

    #[test]
    fn dispatches_hover_action() {
        let registry = LspRegistry::new();
        registry.register("rust", LspServerStatus::Connected, None, vec![]);

        let result = registry
            .dispatch("hover", Some("src/main.rs"), Some(10), Some(5), None)
            .unwrap();
        assert_eq!(result["action"], "hover");
        assert_eq!(result["language"], "rust");
    }

    #[test]
    fn rejects_action_on_disconnected_server() {
        let registry = LspRegistry::new();
        registry.register("rust", LspServerStatus::Disconnected, None, vec![]);

        assert!(registry
            .dispatch("hover", Some("src/main.rs"), Some(1), Some(0), None)
            .is_err());
    }

    #[test]
    fn rejects_unknown_action() {
        let registry = LspRegistry::new();
        assert!(registry
            .dispatch("unknown_action", Some("file.rs"), None, None, None)
            .is_err());
    }

    #[test]
    fn disconnects_server() {
        let registry = LspRegistry::new();
        registry.register("rust", LspServerStatus::Connected, None, vec![]);
        assert_eq!(registry.len(), 1);

        let removed = registry.disconnect("rust");
        assert!(removed.is_some());
        assert!(registry.is_empty());
    }

    #[test]
    fn lsp_action_from_str_all_aliases() {
        // given
        let cases = [
            ("diagnostics", Some(LspAction::Diagnostics)),
            ("health", Some(LspAction::Health)),
            ("status", Some(LspAction::Health)),
            ("hover", Some(LspAction::Hover)),
            ("definition", Some(LspAction::Definition)),
            ("goto_definition", Some(LspAction::Definition)),
            ("references", Some(LspAction::References)),
            ("find_references", Some(LspAction::References)),
            ("completion", Some(LspAction::Completion)),
            ("completions", Some(LspAction::Completion)),
            ("symbols", Some(LspAction::Symbols)),
            ("document_symbols", Some(LspAction::Symbols)),
            ("format", Some(LspAction::Format)),
            ("formatting", Some(LspAction::Format)),
            ("unknown", None),
        ];

        // when
        let resolved: Vec<_> = cases
            .into_iter()
            .map(|(input, expected)| (input, LspAction::from_str(input), expected))
            .collect();

        // then
        for (input, actual, expected) in resolved {
            assert_eq!(actual, expected, "unexpected action resolution for {input}");
        }
    }

    #[test]
    fn lsp_server_status_display_all_variants() {
        // given
        let cases = [
            (LspServerStatus::Connected, "connected"),
            (LspServerStatus::Disconnected, "disconnected"),
            (LspServerStatus::Starting, "starting"),
            (LspServerStatus::Error, "error"),
        ];

        // when
        let rendered: Vec<_> = cases
            .into_iter()
            .map(|(status, expected)| (status.to_string(), expected))
            .collect();

        // then
        assert_eq!(
            rendered,
            vec![
                ("connected".to_string(), "connected"),
                ("disconnected".to_string(), "disconnected"),
                ("starting".to_string(), "starting"),
                ("error".to_string(), "error"),
            ]
        );
    }

    #[test]
    fn dispatch_diagnostics_without_path_aggregates() {
        // given
        let registry = LspRegistry::new();
        registry.register("rust", LspServerStatus::Connected, None, vec![]);
        registry.register("python", LspServerStatus::Connected, None, vec![]);
        registry
            .add_diagnostics(
                "rust",
                vec![LspDiagnostic {
                    path: "src/lib.rs".into(),
                    line: 1,
                    character: 0,
                    severity: "warning".into(),
                    message: "unused import".into(),
                    source: Some("rust-analyzer".into()),
                }],
            )
            .expect("rust diagnostics should add");
        registry
            .add_diagnostics(
                "python",
                vec![LspDiagnostic {
                    path: "script.py".into(),
                    line: 2,
                    character: 4,
                    severity: "error".into(),
                    message: "undefined name".into(),
                    source: Some("pyright".into()),
                }],
            )
            .expect("python diagnostics should add");

        // when
        let result = registry
            .dispatch("diagnostics", None, None, None, None)
            .expect("aggregate diagnostics should work");

        // then
        assert_eq!(result["action"], "diagnostics");
        assert_eq!(result["count"], 2);
        assert_eq!(result["diagnostics"].as_array().map(Vec::len), Some(2));
    }

    #[test]
    fn dispatch_non_diagnostics_requires_path() {
        // given
        let registry = LspRegistry::new();

        // when
        let result = registry.dispatch("hover", None, Some(1), Some(0), None);

        // then
        assert_eq!(
            result.expect_err("path should be required"),
            "path is required for this LSP action"
        );
    }

    #[test]
    fn dispatch_no_server_for_path_errors() {
        // given
        let registry = LspRegistry::new();

        // when
        let result = registry.dispatch("hover", Some("notes.md"), Some(1), Some(0), None);

        // then
        let error = result.expect_err("missing server should fail");
        assert!(error.contains("no LSP server available for path: notes.md"));
    }

    #[test]
    fn dispatch_disconnected_server_error_payload() {
        // given
        let registry = LspRegistry::new();
        registry.register("typescript", LspServerStatus::Disconnected, None, vec![]);

        // when
        let result = registry.dispatch("hover", Some("src/index.ts"), Some(3), Some(2), None);

        // then
        let error = result.expect_err("disconnected server should fail");
        assert!(error.contains("typescript"));
        assert!(error.contains("disconnected"));
    }

    #[test]
    fn find_server_for_all_extensions() {
        // given
        let registry = LspRegistry::new();
        for language in [
            "rust",
            "typescript",
            "javascript",
            "python",
            "go",
            "java",
            "c",
            "cpp",
            "ruby",
            "lua",
        ] {
            registry.register(language, LspServerStatus::Connected, None, vec![]);
        }
        let cases = [
            ("src/main.rs", "rust"),
            ("src/index.ts", "typescript"),
            ("src/view.tsx", "typescript"),
            ("src/app.js", "javascript"),
            ("src/app.jsx", "javascript"),
            ("script.py", "python"),
            ("main.go", "go"),
            ("Main.java", "java"),
            ("native.c", "c"),
            ("native.h", "c"),
            ("native.cpp", "cpp"),
            ("native.hpp", "cpp"),
            ("native.cc", "cpp"),
            ("script.rb", "ruby"),
            ("script.lua", "lua"),
        ];

        // when
        let resolved: Vec<_> = cases
            .into_iter()
            .map(|(path, expected)| {
                (
                    path,
                    registry
                        .find_server_for_path(path)
                        .map(|server| server.language),
                    expected,
                )
            })
            .collect();

        // then
        for (path, actual, expected) in resolved {
            assert_eq!(
                actual.as_deref(),
                Some(expected),
                "unexpected mapping for {path}"
            );
        }
    }

    #[test]
    fn find_server_for_path_no_extension() {
        // given
        let registry = LspRegistry::new();
        registry.register("rust", LspServerStatus::Connected, None, vec![]);

        // when
        let result = registry.find_server_for_path("Makefile");

        // then
        assert!(result.is_none());
    }

    #[test]
    fn list_servers_with_multiple() {
        // given
        let registry = LspRegistry::new();
        registry.register("rust", LspServerStatus::Connected, None, vec![]);
        registry.register("typescript", LspServerStatus::Starting, None, vec![]);
        registry.register("python", LspServerStatus::Error, None, vec![]);

        // when
        let servers = registry.list_servers();

        // then
        assert_eq!(servers.len(), 3);
        assert!(servers.iter().any(|server| server.language == "rust"));
        assert!(servers.iter().any(|server| server.language == "typescript"));
        assert!(servers.iter().any(|server| server.language == "python"));
    }

    #[test]
    fn get_missing_server_returns_none() {
        // given
        let registry = LspRegistry::new();

        // when
        let server = registry.get("missing");

        // then
        assert!(server.is_none());
    }

    #[test]
    fn add_diagnostics_missing_language_errors() {
        // given
        let registry = LspRegistry::new();

        // when
        let result = registry.add_diagnostics("missing", vec![]);

        // then
        let error = result.expect_err("missing language should fail");
        assert!(error.contains("LSP server not found for language: missing"));
    }

    #[test]
    fn get_diagnostics_across_servers() {
        // given
        let registry = LspRegistry::new();
        let shared_path = "shared/file.txt";
        registry.register("rust", LspServerStatus::Connected, None, vec![]);
        registry.register("python", LspServerStatus::Connected, None, vec![]);
        registry
            .add_diagnostics(
                "rust",
                vec![LspDiagnostic {
                    path: shared_path.into(),
                    line: 4,
                    character: 1,
                    severity: "warning".into(),
                    message: "warn".into(),
                    source: None,
                }],
            )
            .expect("rust diagnostics should add");
        registry
            .add_diagnostics(
                "python",
                vec![LspDiagnostic {
                    path: shared_path.into(),
                    line: 8,
                    character: 3,
                    severity: "error".into(),
                    message: "err".into(),
                    source: None,
                }],
            )
            .expect("python diagnostics should add");

        // when
        let diagnostics = registry.get_diagnostics(shared_path);

        // then
        assert_eq!(diagnostics.len(), 2);
        assert!(diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message == "warn"));
        assert!(diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message == "err"));
    }

    #[test]
    fn clear_diagnostics_missing_language_errors() {
        // given
        let registry = LspRegistry::new();

        // when
        let result = registry.clear_diagnostics("missing");

        // then
        let error = result.expect_err("missing language should fail");
        assert!(error.contains("LSP server not found for language: missing"));
    }

    #[test]
    fn health_state_round_trip_blocks_after_reload() {
        // given
        let path = temp_health_path("round-trip");
        let registry = LspRegistry::new();
        registry.register("rust", LspServerStatus::Disconnected, None, vec![]);

        // when
        for _ in 0..3 {
            let _ = registry.dispatch("hover", Some("src/main.rs"), Some(1), Some(0), None);
        }
        let blocked = registry
            .dispatch("hover", Some("src/main.rs"), Some(1), Some(0), None)
            .expect_err("dispatch should be blocked after repeated failures");
        assert!(
            blocked.contains("cooldown"),
            "expected cooldown error, got: {blocked}"
        );
        registry
            .persist_health_to_path(&path)
            .expect("health state should persist");

        let reloaded = LspRegistry::new();
        reloaded
            .load_health_from_path(&path)
            .expect("health state should reload");
        reloaded.register("rust", LspServerStatus::Disconnected, None, vec![]);

        // then
        let after_reload = reloaded
            .dispatch("hover", Some("src/main.rs"), Some(1), Some(0), None)
            .expect_err("reloaded health should preserve cooldown");
        assert!(
            after_reload.contains("cooldown"),
            "expected cooldown after reload, got: {after_reload}"
        );

        cleanup_temp_health_path(&path);
    }

    #[test]
    fn dispatch_health_action_returns_health_snapshot() {
        // given
        let registry = LspRegistry::new();
        registry.register("rust", LspServerStatus::Disconnected, None, vec![]);
        let _ = registry.dispatch("hover", Some("src/main.rs"), Some(1), Some(0), None);

        // when
        let health = registry
            .dispatch("health", None, None, None, None)
            .expect("health action should succeed");

        // then
        assert_eq!(health["action"], "health");
        assert_eq!(health["health"]["rust"]["total_attempts"], 1);
        assert_eq!(health["health"]["rust"]["total_failures"], 1);
    }

    fn temp_health_path(suffix: &str) -> std::path::PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "claw-lsp-health-{}-{}-{}",
            std::process::id(),
            suffix,
            nanos
        ));
        std::fs::create_dir_all(&dir).expect("temp health directory should be created");
        dir.join("lsp_health_state.json")
    }

    fn cleanup_temp_health_path(path: &std::path::Path) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::remove_dir_all(parent);
        }
    }
}
