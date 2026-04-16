use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::ApiError;
use crate::http_client::build_http_client_or_default;
use crate::types::{
    ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent,
    InputContentBlock, InputMessage, MessageDelta, MessageDeltaEvent, MessageRequest,
    MessageResponse, MessageStartEvent, MessageStopEvent, OutputContentBlock, StreamEvent,
    ToolChoice, ToolDefinition, ToolResultContentBlock, Usage,
};

use super::{preflight_message_request, Provider, ProviderFuture};

pub const DEFAULT_XAI_BASE_URL: &str = "https://api.x.ai/v1";
pub const DEFAULT_OPENAI_BASE_URL: &str = "http://127.0.0.1:8080/v1";
pub const DEFAULT_DASHSCOPE_BASE_URL: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1";
const REQUEST_ID_HEADER: &str = "request-id";
const ALT_REQUEST_ID_HEADER: &str = "x-request-id";
const DEFAULT_INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const DEFAULT_MAX_BACKOFF: Duration = Duration::from_secs(128);
const DEFAULT_MAX_RETRIES: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenAiCompatConfig {
    pub provider_name: &'static str,
    pub api_key_env: &'static str,
    pub base_url_env: &'static str,
    pub default_base_url: &'static str,
}

const XAI_ENV_VARS: &[&str] = &["XAI_API_KEY"];
const OPENAI_ENV_VARS: &[&str] = &["OPENAI_API_KEY"];
const DASHSCOPE_ENV_VARS: &[&str] = &["DASHSCOPE_API_KEY"];

impl OpenAiCompatConfig {
    #[must_use]
    pub const fn xai() -> Self {
        Self {
            provider_name: "xAI",
            api_key_env: "XAI_API_KEY",
            base_url_env: "XAI_BASE_URL",
            default_base_url: DEFAULT_XAI_BASE_URL,
        }
    }

    #[must_use]
    pub const fn openai() -> Self {
        Self {
            provider_name: "OpenAI",
            api_key_env: "OPENAI_API_KEY",
            base_url_env: "OPENAI_BASE_URL",
            default_base_url: DEFAULT_OPENAI_BASE_URL,
        }
    }

    /// Alibaba DashScope compatible-mode endpoint (Qwen family models).
    /// Uses the OpenAI-compatible REST shape at /compatible-mode/v1.
    /// Requested via Discord #clawcode-get-help: native Alibaba API for
    /// higher rate limits than going through OpenRouter.
    #[must_use]
    pub const fn dashscope() -> Self {
        Self {
            provider_name: "DashScope",
            api_key_env: "DASHSCOPE_API_KEY",
            base_url_env: "DASHSCOPE_BASE_URL",
            default_base_url: DEFAULT_DASHSCOPE_BASE_URL,
        }
    }

    #[must_use]
    pub fn credential_env_vars(self) -> &'static [&'static str] {
        match self.provider_name {
            "xAI" => XAI_ENV_VARS,
            "OpenAI" => OPENAI_ENV_VARS,
            "DashScope" => DASHSCOPE_ENV_VARS,
            _ => &[],
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatClient {
    http: reqwest::Client,
    api_key: String,
    config: OpenAiCompatConfig,
    base_url: String,
    max_retries: u32,
    initial_backoff: Duration,
    max_backoff: Duration,
}

impl OpenAiCompatClient {
    const fn config(&self) -> OpenAiCompatConfig {
        self.config
    }

    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
    #[must_use]
    pub fn new(api_key: impl Into<String>, config: OpenAiCompatConfig) -> Self {
        Self {
            http: build_http_client_or_default(),
            api_key: api_key.into(),
            config,
            base_url: read_base_url(config),
            max_retries: DEFAULT_MAX_RETRIES,
            initial_backoff: DEFAULT_INITIAL_BACKOFF,
            max_backoff: DEFAULT_MAX_BACKOFF,
        }
    }

    pub fn from_env(config: OpenAiCompatConfig) -> Result<Self, ApiError> {
        let base_url = read_base_url(config);
        let api_key = read_env_non_empty(config.api_key_env)?;
        if api_key.is_none() && !(config.provider_name == "OpenAI" && is_local_base_url(&base_url))
        {
            return Err(ApiError::missing_credentials(
                config.provider_name,
                config.credential_env_vars(),
            ));
        }
        Ok(Self::new(api_key.unwrap_or_default(), config))
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    #[must_use]
    pub fn with_retry_policy(
        mut self,
        max_retries: u32,
        initial_backoff: Duration,
        max_backoff: Duration,
    ) -> Self {
        self.max_retries = max_retries;
        self.initial_backoff = initial_backoff;
        self.max_backoff = max_backoff;
        self
    }

    pub async fn send_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        let request = MessageRequest {
            stream: false,
            ..request.clone()
        };
        preflight_message_request(&request)?;
        let response = self.send_with_retry(&request).await?;
        let request_id = request_id_from_headers(response.headers());
        let body = response.text().await.map_err(ApiError::from)?;
        let payload = serde_json::from_str::<ChatCompletionResponse>(&body).map_err(|error| {
            ApiError::json_deserialize(self.config.provider_name, &request.model, &body, error)
        })?;
        let mut normalized = normalize_response(&request.model, payload)?;
        if normalized.request_id.is_none() {
            normalized.request_id = request_id;
        }
        Ok(normalized)
    }

    pub async fn stream_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageStream, ApiError> {
        preflight_message_request(request)?;
        let response = self
            .send_with_retry(&request.clone().with_streaming())
            .await?;
        Ok(MessageStream {
            request_id: request_id_from_headers(response.headers()),
            response,
            parser: OpenAiSseParser::with_context(self.config.provider_name, request.model.clone()),
            pending: VecDeque::new(),
            done: false,
            state: StreamState::new(request.model.clone()),
        })
    }

    async fn send_with_retry(
        &self,
        request: &MessageRequest,
    ) -> Result<reqwest::Response, ApiError> {
        let mut attempts = 0;

        let last_error = loop {
            attempts += 1;
            let retryable_error = match self.send_raw_request(request).await {
                Ok(response) => match expect_success(response).await {
                    Ok(response) => return Ok(response),
                    Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => error,
                    Err(error) => return Err(error),
                },
                Err(error) if error.is_retryable() && attempts <= self.max_retries + 1 => error,
                Err(error) => return Err(error),
            };

            if attempts > self.max_retries {
                break retryable_error;
            }

            tokio::time::sleep(self.jittered_backoff_for_attempt(attempts)?).await;
        };

        Err(ApiError::RetriesExhausted {
            attempts,
            last_error: Box::new(last_error),
        })
    }

    async fn send_raw_request(
        &self,
        request: &MessageRequest,
    ) -> Result<reqwest::Response, ApiError> {
        let request_url = chat_completions_endpoint(&self.base_url);
        let mut builder = self
            .http
            .post(&request_url)
            .header("content-type", "application/json")
            .json(&build_chat_completion_request_with_base_url(
                request,
                self.config(),
                &self.base_url,
            ));
        if !self.api_key.trim().is_empty() {
            builder = builder.bearer_auth(&self.api_key);
        }
        builder.send().await.map_err(ApiError::from)
    }

    fn backoff_for_attempt(&self, attempt: u32) -> Result<Duration, ApiError> {
        let Some(multiplier) = 1_u32.checked_shl(attempt.saturating_sub(1)) else {
            return Err(ApiError::BackoffOverflow {
                attempt,
                base_delay: self.initial_backoff,
            });
        };
        Ok(self
            .initial_backoff
            .checked_mul(multiplier)
            .map_or(self.max_backoff, |delay| delay.min(self.max_backoff)))
    }

    fn jittered_backoff_for_attempt(&self, attempt: u32) -> Result<Duration, ApiError> {
        let base = self.backoff_for_attempt(attempt)?;
        Ok(base + jitter_for_base(base))
    }
}

/// Process-wide counter that guarantees distinct jitter samples even when
/// the system clock resolution is coarser than consecutive retry sleeps.
static JITTER_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Returns a random additive jitter in `[0, base]` to decorrelate retries
/// from multiple concurrent clients. Entropy is drawn from the nanosecond
/// wall clock mixed with a monotonic counter and run through a splitmix64
/// finalizer; adequate for retry jitter (no cryptographic requirement).
fn jitter_for_base(base: Duration) -> Duration {
    let base_nanos = u64::try_from(base.as_nanos()).unwrap_or(u64::MAX);
    if base_nanos == 0 {
        return Duration::ZERO;
    }
    let raw_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0);
    let tick = JITTER_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut mixed = raw_nanos
        .wrapping_add(tick)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^= mixed >> 31;
    let jitter_nanos = mixed % base_nanos.saturating_add(1);
    Duration::from_nanos(jitter_nanos)
}

impl Provider for OpenAiCompatClient {
    type Stream = MessageStream;

    fn send_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, MessageResponse> {
        Box::pin(async move { self.send_message(request).await })
    }

    fn stream_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, Self::Stream> {
        Box::pin(async move { self.stream_message(request).await })
    }
}

#[derive(Debug)]
pub struct MessageStream {
    request_id: Option<String>,
    response: reqwest::Response,
    parser: OpenAiSseParser,
    pending: VecDeque<StreamEvent>,
    done: bool,
    state: StreamState,
}

impl MessageStream {
    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }

            if self.done {
                self.pending.extend(self.state.finish()?);
                if let Some(event) = self.pending.pop_front() {
                    return Ok(Some(event));
                }
                return Ok(None);
            }

            match self.response.chunk().await? {
                Some(chunk) => {
                    for parsed in self.parser.push(&chunk)? {
                        self.pending.extend(self.state.ingest_chunk(parsed)?);
                    }
                }
                None => {
                    self.done = true;
                }
            }
        }
    }
}

#[derive(Debug, Default)]
struct OpenAiSseParser {
    buffer: Vec<u8>,
    provider: String,
    model: String,
}

impl OpenAiSseParser {
    fn with_context(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            buffer: Vec::new(),
            provider: provider.into(),
            model: model.into(),
        }
    }

    fn push(&mut self, chunk: &[u8]) -> Result<Vec<ChatCompletionChunk>, ApiError> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();

        while let Some(frame) = next_sse_frame(&mut self.buffer) {
            if let Some(event) = parse_sse_frame(&frame, &self.provider, &self.model)? {
                events.push(event);
            }
        }

        Ok(events)
    }
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug)]
struct StreamState {
    model: String,
    qwen_model: bool,
    message_started: bool,
    text_started: bool,
    text_finished: bool,
    finished: bool,
    stop_reason: Option<String>,
    usage: Option<Usage>,
    tool_calls: BTreeMap<u32, ToolCallState>,
    buffered_text: String,
}

impl StreamState {
    fn new(model: String) -> Self {
        let qwen_model = is_qwen_family_model(&model);
        Self {
            model,
            qwen_model,
            message_started: false,
            text_started: false,
            text_finished: false,
            finished: false,
            stop_reason: None,
            usage: None,
            tool_calls: BTreeMap::new(),
            buffered_text: String::new(),
        }
    }

    fn ingest_chunk(&mut self, chunk: ChatCompletionChunk) -> Result<Vec<StreamEvent>, ApiError> {
        let mut events = Vec::new();
        if !self.message_started {
            self.message_started = true;
            events.push(StreamEvent::MessageStart(MessageStartEvent {
                message: MessageResponse {
                    id: chunk.id.clone(),
                    kind: "message".to_string(),
                    role: "assistant".to_string(),
                    content: Vec::new(),
                    model: chunk.model.clone().unwrap_or_else(|| self.model.clone()),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: Usage {
                        input_tokens: 0,
                        cache_creation_input_tokens: 0,
                        cache_read_input_tokens: 0,
                        output_tokens: 0,
                    },
                    request_id: None,
                },
            }));
        }

        if let Some(usage) = chunk.usage {
            self.usage = Some(Usage {
                input_tokens: usage.prompt_tokens,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                output_tokens: usage.completion_tokens,
            });
        }

        for choice in chunk.choices {
            if let Some(content) = delta_text_content(&choice.delta) {
                if self.qwen_model {
                    // Qwen3.5 local runtimes frequently emit textual
                    // `<tool_call>` tags in content. Buffer text until finish so
                    // we can promote those tags into executable tool-use blocks.
                    self.buffered_text.push_str(&content);
                } else {
                    if !self.text_started {
                        self.text_started = true;
                        events.push(StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                            index: 0,
                            content_block: OutputContentBlock::Text {
                                text: String::new(),
                            },
                        }));
                    }
                    events.push(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                        index: 0,
                        delta: ContentBlockDelta::TextDelta { text: content },
                    }));
                }
            }

            let mut delta_tool_calls = choice.delta.tool_calls;
            if delta_tool_calls.is_empty() {
                if let Some(function_call) = choice.delta.function_call {
                    delta_tool_calls.push(DeltaToolCall {
                        index: 0,
                        id: None,
                        function: function_call,
                    });
                }
            }

            for tool_call in delta_tool_calls {
                let state = self.tool_calls.entry(tool_call.index).or_default();
                state.apply(tool_call);
                let block_index = state.block_index();
                if !state.started {
                    if let Some(start_event) = state.start_event()? {
                        state.started = true;
                        events.push(StreamEvent::ContentBlockStart(start_event));
                    } else {
                        continue;
                    }
                }
                if let Some(delta_event) = state.delta_event() {
                    events.push(StreamEvent::ContentBlockDelta(delta_event));
                }
                if choice.finish_reason.as_deref() == Some("tool_calls") && !state.stopped {
                    state.stopped = true;
                    events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                        index: block_index,
                    }));
                }
            }

            if let Some(finish_reason) = choice.finish_reason {
                self.stop_reason = Some(normalize_finish_reason(&finish_reason));
                if finish_reason == "tool_calls" {
                    for state in self.tool_calls.values_mut() {
                        if state.started && !state.stopped {
                            state.stopped = true;
                            events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                                index: state.block_index(),
                            }));
                        }
                    }
                }
            }
        }

        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, ApiError> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;

        let mut events = Vec::new();
        if self.qwen_model {
            let mut inferred_tool_calls = Vec::new();
            let mut final_text = self.buffered_text.clone();
            if self.tool_calls.is_empty() {
                let parsed = parse_qwen_textual_tool_calls(&self.buffered_text);
                final_text = parsed.remaining_text;
                inferred_tool_calls = parsed.calls;
            }

            if !final_text.is_empty() {
                self.text_started = true;
                self.text_finished = true;
                events.push(StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                    index: 0,
                    content_block: OutputContentBlock::Text {
                        text: String::new(),
                    },
                }));
                events.push(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                    index: 0,
                    delta: ContentBlockDelta::TextDelta { text: final_text },
                }));
                events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                    index: 0,
                }));
            }

            if !inferred_tool_calls.is_empty() {
                let mut next_index = self
                    .tool_calls
                    .keys()
                    .max()
                    .copied()
                    .unwrap_or(0)
                    .saturating_add(1);
                for call in inferred_tool_calls {
                    let arguments = call.function.arguments;
                    events.push(StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                        index: next_index,
                        content_block: OutputContentBlock::ToolUse {
                            id: call.id,
                            name: call.function.name,
                            input: json!({}),
                        },
                    }));
                    if !arguments.is_empty() {
                        events.push(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                            index: next_index,
                            delta: ContentBlockDelta::InputJsonDelta {
                                partial_json: arguments,
                            },
                        }));
                    }
                    events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                        index: next_index,
                    }));
                    next_index = next_index.saturating_add(1);
                }
                if matches!(self.stop_reason.as_deref(), None | Some("end_turn")) {
                    self.stop_reason = Some("tool_use".to_string());
                }
            }
        } else if self.text_started && !self.text_finished {
            self.text_finished = true;
            events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                index: 0,
            }));
        }

        for state in self.tool_calls.values_mut() {
            if !state.started {
                if let Some(start_event) = state.start_event()? {
                    state.started = true;
                    events.push(StreamEvent::ContentBlockStart(start_event));
                    if let Some(delta_event) = state.delta_event() {
                        events.push(StreamEvent::ContentBlockDelta(delta_event));
                    }
                }
            }
            if state.started && !state.stopped {
                state.stopped = true;
                events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                    index: state.block_index(),
                }));
            }
        }

        if self.message_started {
            events.push(StreamEvent::MessageDelta(MessageDeltaEvent {
                delta: MessageDelta {
                    stop_reason: Some(
                        self.stop_reason
                            .clone()
                            .unwrap_or_else(|| "end_turn".to_string()),
                    ),
                    stop_sequence: None,
                },
                usage: self.usage.clone().unwrap_or(Usage {
                    input_tokens: 0,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                    output_tokens: 0,
                }),
            }));
            events.push(StreamEvent::MessageStop(MessageStopEvent {}));
        }
        Ok(events)
    }
}

#[derive(Debug, Default)]
struct ToolCallState {
    openai_index: u32,
    id: Option<String>,
    name: Option<String>,
    arguments: String,
    emitted_len: usize,
    started: bool,
    stopped: bool,
}

impl ToolCallState {
    fn apply(&mut self, tool_call: DeltaToolCall) {
        self.openai_index = tool_call.index;
        if let Some(id) = tool_call.id {
            self.id = Some(id);
        }
        if let Some(name) = tool_call.function.name {
            self.name = Some(name);
        }
        if let Some(arguments) = tool_call.function.arguments {
            self.arguments.push_str(&arguments);
        }
    }

    const fn block_index(&self) -> u32 {
        self.openai_index + 1
    }

    #[allow(clippy::unnecessary_wraps)]
    fn start_event(&self) -> Result<Option<ContentBlockStartEvent>, ApiError> {
        let Some(name) = self.name.clone() else {
            return Ok(None);
        };
        let id = self
            .id
            .clone()
            .unwrap_or_else(|| format!("tool_call_{}", self.openai_index));
        Ok(Some(ContentBlockStartEvent {
            index: self.block_index(),
            content_block: OutputContentBlock::ToolUse {
                id,
                name,
                input: json!({}),
            },
        }))
    }

    fn delta_event(&mut self) -> Option<ContentBlockDeltaEvent> {
        if self.emitted_len >= self.arguments.len() {
            return None;
        }
        let delta = self.arguments[self.emitted_len..].to_string();
        self.emitted_len = self.arguments.len();
        Some(ContentBlockDeltaEvent {
            index: self.block_index(),
            delta: ContentBlockDelta::InputJsonDelta {
                partial_json: delta,
            },
        })
    }
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    id: String,
    model: String,
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<Value>,
    #[serde(default)]
    tool_calls: Vec<ResponseToolCall>,
    #[serde(default)]
    function_call: Option<ResponseToolFunction>,
}

#[derive(Debug, Clone, Deserialize)]
struct ResponseToolCall {
    id: String,
    function: ResponseToolFunction,
}

#[derive(Debug, Clone, Deserialize)]
struct ResponseToolFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    id: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    delta: ChunkDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ChunkDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<Value>,
    #[serde(default)]
    tool_calls: Vec<DeltaToolCall>,
    #[serde(default)]
    function_call: Option<DeltaFunction>,
}

#[derive(Debug, Deserialize)]
struct DeltaToolCall {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: DeltaFunction,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct DeltaFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    #[serde(rename = "type")]
    error_type: Option<String>,
    message: Option<String>,
}

/// Returns true for models known to reject tuning parameters like temperature,
/// top_p, frequency_penalty, and presence_penalty. These are typically
/// reasoning/chain-of-thought models with fixed sampling.
fn is_reasoning_model(model: &str) -> bool {
    let lowered = model.to_ascii_lowercase();
    // Strip any provider/ prefix for the check (e.g. qwen/qwen-qwq -> qwen-qwq)
    let canonical = lowered.rsplit('/').next().unwrap_or(lowered.as_str());
    // OpenAI reasoning models
    canonical.starts_with("o1")
        || canonical.starts_with("o3")
        || canonical.starts_with("o4")
        // xAI reasoning: grok-3-mini always uses reasoning mode
        || canonical == "grok-3-mini"
        // Alibaba DashScope reasoning variants (QwQ + Qwen3-Thinking family)
        || canonical.starts_with("qwen-qwq")
        || canonical.starts_with("qwq")
        || canonical.contains("thinking")
}

fn is_qwen_family_model(model: &str) -> bool {
    let lowered = model.to_ascii_lowercase();
    let canonical = lowered.rsplit('/').next().unwrap_or(lowered.as_str());
    canonical.starts_with("qwen") || canonical.starts_with("qwq") || canonical.contains("qwen3")
}

/// Strip routing prefix (e.g., "openai/gpt-4" → "gpt-4") for the wire.
/// The prefix is used only to select transport; the backend expects the
/// bare model id.
fn strip_routing_prefix(model: &str) -> &str {
    if let Some(pos) = model.find('/') {
        let prefix = &model[..pos];
        // Only strip if the prefix before "/" is a known routing prefix,
        // not if "/" appears in the middle of the model name for other reasons.
        if matches!(prefix, "openai" | "xai" | "grok" | "qwen") {
            &model[pos + 1..]
        } else {
            model
        }
    } else {
        model
    }
}

#[cfg(test)]
fn build_chat_completion_request(request: &MessageRequest, config: OpenAiCompatConfig) -> Value {
    build_chat_completion_request_with_base_url(request, config, config.default_base_url)
}

fn build_chat_completion_request_with_base_url(
    request: &MessageRequest,
    config: OpenAiCompatConfig,
    base_url: &str,
) -> Value {
    let mut messages = Vec::new();
    if let Some(system) = request.system.as_ref().filter(|value| !value.is_empty()) {
        messages.push(json!({
            "role": "system",
            "content": system,
        }));
    }
    for message in &request.messages {
        messages.extend(translate_message(message));
    }

    // Strip routing prefix (e.g., "openai/gpt-4" → "gpt-4") for the wire.
    let wire_model = strip_routing_prefix(&request.model);

    // gpt-5* requires `max_completion_tokens`; older OpenAI models accept both.
    // We send the correct field based on the wire model name so gpt-5.x requests
    // don't fail with "unknown field max_tokens".
    let max_tokens_key = if wire_model.starts_with("gpt-5") {
        "max_completion_tokens"
    } else {
        "max_tokens"
    };

    let mut payload = json!({
        "model": wire_model,
        max_tokens_key: request.max_tokens,
        "messages": messages,
        "stream": request.stream,
    });

    if request.stream && should_request_stream_usage(config) {
        payload["stream_options"] = json!({ "include_usage": true });
    }

    if let Some(tools) = &request.tools {
        payload["tools"] =
            Value::Array(tools.iter().map(openai_tool_definition).collect::<Vec<_>>());
    }
    if let Some(tool_choice) = &request.tool_choice {
        payload["tool_choice"] = openai_tool_choice(tool_choice);
    }

    // OpenAI-compatible tuning parameters — only included when explicitly set.
    // Reasoning models (o1/o3/o4/grok-3-mini) reject these params with 400;
    // silently strip them to avoid cryptic provider errors.
    if !is_reasoning_model(&request.model) {
        if let Some(temperature) = request.temperature {
            payload["temperature"] = json!(temperature);
        }
        if let Some(top_p) = request.top_p {
            payload["top_p"] = json!(top_p);
        }
        if let Some(frequency_penalty) = request.frequency_penalty {
            payload["frequency_penalty"] = json!(frequency_penalty);
        }
        if let Some(presence_penalty) = request.presence_penalty {
            payload["presence_penalty"] = json!(presence_penalty);
        }
    }
    // stop is generally safe for all providers
    if let Some(stop) = &request.stop {
        if !stop.is_empty() {
            payload["stop"] = json!(stop);
        }
    }
    // reasoning_effort for OpenAI-compatible reasoning models (o4-mini, o3, etc.)
    if let Some(effort) = &request.reasoning_effort {
        payload["reasoning_effort"] = json!(effort);
    }
    // Local-only extension payload fields used by llama.cpp-style runtimes.
    // Never emitted for non-local or non-Qwen routes so provider-standard
    // OpenAI/xAI/DashScope payloads remain spec-clean.
    if should_emit_local_qwen_extensions(config, base_url, &request.model) {
        if let Some(top_k) = request.top_k {
            payload["top_k"] = json!(top_k);
        }
        if let Some(min_p) = request.min_p {
            payload["min_p"] = json!(min_p);
        }
        if let Some(repeat_penalty) = request.repeat_penalty {
            payload["repeat_penalty"] = json!(repeat_penalty);
        }
        if let Some(repeat_last_n) = request.repeat_last_n {
            payload["repeat_last_n"] = json!(repeat_last_n);
        }
        if let Some(kwargs) = &request.chat_template_kwargs {
            payload["chat_template_kwargs"] = kwargs.clone();
        }
    }

    payload
}

fn should_emit_local_qwen_extensions(
    config: OpenAiCompatConfig,
    base_url: &str,
    model: &str,
) -> bool {
    config.provider_name == "OpenAI" && is_local_base_url(base_url) && is_qwen_family_model(model)
}

fn translate_message(message: &InputMessage) -> Vec<Value> {
    match message.role.as_str() {
        "assistant" => {
            let mut text = String::new();
            let mut tool_calls = Vec::new();
            for block in &message.content {
                match block {
                    InputContentBlock::Text { text: value } => text.push_str(value),
                    InputContentBlock::ToolUse { id, name, input } => tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": input.to_string(),
                        }
                    })),
                    InputContentBlock::ToolResult { .. } => {}
                }
            }
            if text.is_empty() && tool_calls.is_empty() {
                Vec::new()
            } else {
                vec![json!({
                    "role": "assistant",
                    "content": (!text.is_empty()).then_some(text),
                    "tool_calls": tool_calls,
                })]
            }
        }
        _ => message
            .content
            .iter()
            .filter_map(|block| match block {
                InputContentBlock::Text { text } => Some(json!({
                    "role": "user",
                    "content": text,
                })),
                InputContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => Some(json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": flatten_tool_result_content(content),
                    "is_error": is_error,
                })),
                InputContentBlock::ToolUse { .. } => None,
            })
            .collect(),
    }
}

fn flatten_tool_result_content(content: &[ToolResultContentBlock]) -> String {
    content
        .iter()
        .map(|block| match block {
            ToolResultContentBlock::Text { text } => text.clone(),
            ToolResultContentBlock::Json { value } => value.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Recursively ensure every object-type node in a JSON Schema has
/// `"properties"` (at least `{}`) and `"additionalProperties": false`.
/// The OpenAI `/responses` endpoint validates schemas strictly and rejects
/// objects that omit these fields; `/chat/completions` is lenient but also
/// accepts them, so we normalise unconditionally.
fn normalize_object_schema(schema: &mut Value) {
    if let Some(obj) = schema.as_object_mut() {
        if obj.get("type").and_then(Value::as_str) == Some("object") {
            obj.entry("properties").or_insert_with(|| json!({}));
            obj.entry("additionalProperties")
                .or_insert(Value::Bool(false));
        }
        // Recurse into properties values
        if let Some(props) = obj.get_mut("properties") {
            if let Some(props_obj) = props.as_object_mut() {
                let keys: Vec<String> = props_obj.keys().cloned().collect();
                for k in keys {
                    if let Some(v) = props_obj.get_mut(&k) {
                        normalize_object_schema(v);
                    }
                }
            }
        }
        // Recurse into items (arrays)
        if let Some(items) = obj.get_mut("items") {
            normalize_object_schema(items);
        }
    }
}

fn openai_tool_definition(tool: &ToolDefinition) -> Value {
    let mut parameters = tool.input_schema.clone();
    normalize_object_schema(&mut parameters);
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        }
    })
}

fn openai_tool_choice(tool_choice: &ToolChoice) -> Value {
    match tool_choice {
        ToolChoice::Auto => Value::String("auto".to_string()),
        ToolChoice::Any => Value::String("required".to_string()),
        ToolChoice::Tool { name } => json!({
            "type": "function",
            "function": { "name": name },
        }),
    }
}

fn should_request_stream_usage(config: OpenAiCompatConfig) -> bool {
    matches!(config.provider_name, "OpenAI")
}

fn normalize_response(
    model: &str,
    response: ChatCompletionResponse,
) -> Result<MessageResponse, ApiError> {
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or(ApiError::InvalidSseFrame(
            "chat completion response missing choices",
        ))?;
    let mut native_tool_calls = choice.message.tool_calls.clone();
    if native_tool_calls.is_empty() {
        if let Some(function_call) = choice.message.function_call.clone() {
            native_tool_calls.push(ResponseToolCall {
                id: "tool_call_0".to_string(),
                function: function_call,
            });
        }
    }

    let mut content = Vec::new();
    let mut inferred_tool_calls = Vec::new();
    let mut text_content = message_text_content(&choice.message);
    if native_tool_calls.is_empty() && is_qwen_family_model(model) {
        if let Some(text) = text_content.as_ref() {
            let parsed = parse_qwen_textual_tool_calls(text);
            inferred_tool_calls = parsed.calls;
            text_content = if parsed.remaining_text.is_empty() {
                None
            } else {
                Some(parsed.remaining_text)
            };
        }
    }

    if let Some(text) = text_content {
        content.push(OutputContentBlock::Text { text });
    }
    for tool_call in native_tool_calls.into_iter().chain(inferred_tool_calls) {
        content.push(OutputContentBlock::ToolUse {
            id: tool_call.id,
            name: tool_call.function.name,
            input: parse_tool_arguments(&tool_call.function.arguments),
        });
    }

    let mut stop_reason = choice
        .finish_reason
        .map(|value| normalize_finish_reason(&value));
    if stop_reason.as_deref() == Some("end_turn")
        && content
            .iter()
            .any(|block| matches!(block, OutputContentBlock::ToolUse { .. }))
    {
        stop_reason = Some("tool_use".to_string());
    }

    Ok(MessageResponse {
        id: response.id,
        kind: "message".to_string(),
        role: choice.message.role,
        content,
        model: response.model.if_empty_then(model.to_string()),
        stop_reason,
        stop_sequence: None,
        usage: Usage {
            input_tokens: response
                .usage
                .as_ref()
                .map_or(0, |usage| usage.prompt_tokens),
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            output_tokens: response
                .usage
                .as_ref()
                .map_or(0, |usage| usage.completion_tokens),
        },
        request_id: None,
    })
}

fn parse_tool_arguments(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| json!({ "raw": arguments }))
}

#[derive(Debug, Default)]
struct QwenParsedToolCalls {
    remaining_text: String,
    calls: Vec<ResponseToolCall>,
}

fn parse_qwen_textual_tool_calls(text: &str) -> QwenParsedToolCalls {
    let parsed = parse_qwen_xml_tool_calls(text);
    let parsed = parse_qwen_tool_name_tag_calls(&parsed.remaining_text, parsed.calls);
    let parsed = parse_qwen_tool_name_and_args_tag_calls(&parsed.remaining_text, parsed.calls);
    let parsed = parse_qwen_loose_tool_call_tags(&parsed.remaining_text, parsed.calls);
    let parsed = parse_qwen_markdown_command_calls(&parsed.remaining_text, parsed.calls);
    parse_qwen_inline_backtick_tool_calls(&parsed.remaining_text, parsed.calls)
}

fn parse_qwen_xml_tool_calls(text: &str) -> QwenParsedToolCalls {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";

    let mut remaining = String::new();
    let mut calls = Vec::new();
    let mut cursor = 0usize;

    while let Some(open_rel) = text[cursor..].find(OPEN) {
        let open_idx = cursor + open_rel;
        remaining.push_str(&text[cursor..open_idx]);
        let block_start = open_idx + OPEN.len();
        let Some(close_rel) = text[block_start..].find(CLOSE) else {
            remaining.push_str(&text[open_idx..]);
            cursor = text.len();
            break;
        };
        let block_end = block_start + close_rel;
        let block = &text[block_start..block_end];
        if let Some(call) = parse_qwen_textual_tool_call_block(block, calls.len()) {
            calls.push(call);
        } else {
            remaining.push_str(&text[open_idx..block_end + CLOSE.len()]);
        }
        cursor = block_end + CLOSE.len();
    }
    if cursor < text.len() {
        remaining.push_str(&text[cursor..]);
    }

    QwenParsedToolCalls {
        remaining_text: remaining,
        calls,
    }
}

fn parse_qwen_loose_tool_call_tags(
    text: &str,
    mut calls: Vec<ResponseToolCall>,
) -> QwenParsedToolCalls {
    const OPEN: &str = "<tool_call>";

    let mut remaining = String::new();
    let mut cursor = 0usize;

    while let Some(open_rel) = text[cursor..].find(OPEN) {
        let open_idx = cursor + open_rel;
        remaining.push_str(&text[cursor..open_idx]);
        let block_start = open_idx + OPEN.len();
        let Some(close_rel) = text[block_start..].find("</") else {
            remaining.push_str(&text[open_idx..]);
            cursor = text.len();
            break;
        };
        let close_start = block_start + close_rel;
        let Some(close_end_rel) = text[close_start..].find('>') else {
            remaining.push_str(&text[open_idx..]);
            cursor = text.len();
            break;
        };
        let close_end = close_start + close_end_rel + 1;
        let block = &text[block_start..close_start];
        if let Some(call) = parse_qwen_flat_tool_call(block, calls.len()) {
            calls.push(call);
        } else {
            remaining.push_str(&text[open_idx..close_end]);
        }
        cursor = close_end;
    }

    if cursor < text.len() {
        remaining.push_str(&text[cursor..]);
    }

    QwenParsedToolCalls {
        remaining_text: remaining,
        calls,
    }
}

fn parse_qwen_tool_name_tag_calls(
    text: &str,
    mut calls: Vec<ResponseToolCall>,
) -> QwenParsedToolCalls {
    const TOOL_OPEN: &str = "<tool>";
    const TOOL_CLOSE: &str = "</tool>";
    let mut remaining = String::new();
    let mut cursor = 0usize;

    while let Some(open_rel) = text[cursor..].find(TOOL_OPEN) {
        let open_idx = cursor + open_rel;
        remaining.push_str(&text[cursor..open_idx]);
        let inner_start = open_idx + TOOL_OPEN.len();
        let Some(close_rel) = text[inner_start..].find(TOOL_CLOSE) else {
            remaining.push_str(&text[open_idx..]);
            cursor = text.len();
            break;
        };
        let inner_end = inner_start + close_rel;
        let close_end = inner_end + TOOL_CLOSE.len();
        let inner = &text[inner_start..inner_end];
        if let Some(call) = parse_qwen_named_tool_tag_block(inner, calls.len()) {
            calls.push(call);
        } else {
            remaining.push_str(&text[open_idx..close_end]);
        }
        cursor = close_end;
    }

    if cursor < text.len() {
        remaining.push_str(&text[cursor..]);
    }

    QwenParsedToolCalls {
        remaining_text: remaining,
        calls,
    }
}

fn parse_qwen_tool_name_and_args_tag_calls(
    text: &str,
    mut calls: Vec<ResponseToolCall>,
) -> QwenParsedToolCalls {
    const OPEN: &str = "<tool_call>";

    let mut remaining = String::new();
    let mut cursor = 0usize;

    while let Some(open_rel) = text[cursor..].find(OPEN) {
        let open_idx = cursor + open_rel;
        remaining.push_str(&text[cursor..open_idx]);
        let block_start = open_idx + OPEN.len();
        let close_tool_use = text[block_start..].find("</tool_use>");
        let close_tool_call = text[block_start..].find("</tool_call>");
        let close_rel = match (close_tool_use, close_tool_call) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        let Some(close_rel) = close_rel else {
            remaining.push_str(&text[open_idx..]);
            cursor = text.len();
            break;
        };
        let close_start = block_start + close_rel;
        let close_end = text[close_start..]
            .find('>')
            .map_or(text.len(), |end_rel| close_start + end_rel + 1);
        let block = &text[block_start..close_start];
        if let Some(call) = parse_qwen_tool_name_and_args_tag_block(block, calls.len()) {
            calls.push(call);
        } else {
            remaining.push_str(&text[open_idx..close_end]);
        }
        cursor = close_end;
    }

    if cursor < text.len() {
        remaining.push_str(&text[cursor..]);
    }

    QwenParsedToolCalls {
        remaining_text: remaining,
        calls,
    }
}

fn parse_qwen_markdown_command_calls(
    text: &str,
    mut calls: Vec<ResponseToolCall>,
) -> QwenParsedToolCalls {
    const OPEN_FENCE: &str = "```";

    let mut remaining = String::new();
    let mut cursor = 0usize;
    while let Some(open_rel) = text[cursor..].find(OPEN_FENCE) {
        let open_idx = cursor + open_rel;
        remaining.push_str(&text[cursor..open_idx]);
        let fence_start = open_idx + OPEN_FENCE.len();
        let Some(close_rel) = text[fence_start..].find(OPEN_FENCE) else {
            remaining.push_str(&text[open_idx..]);
            cursor = text.len();
            break;
        };
        let fence_end = fence_start + close_rel;
        let block = &text[fence_start..fence_end];
        if let Some(call) = parse_qwen_fenced_bash_call(block, calls.len()) {
            calls.push(call);
        } else {
            remaining.push_str(&text[open_idx..fence_end + OPEN_FENCE.len()]);
        }
        cursor = fence_end + OPEN_FENCE.len();
    }
    if cursor < text.len() {
        remaining.push_str(&text[cursor..]);
    }

    QwenParsedToolCalls {
        remaining_text: remaining.trim().to_string(),
        calls,
    }
}

fn parse_qwen_inline_backtick_tool_calls(
    text: &str,
    mut calls: Vec<ResponseToolCall>,
) -> QwenParsedToolCalls {
    let mut remaining = String::new();
    let mut cursor = 0usize;

    while let Some(open_rel) = text[cursor..].find('`') {
        let open_idx = cursor + open_rel;
        remaining.push_str(&text[cursor..open_idx]);
        let inline_start = open_idx + 1;
        let Some(close_rel) = text[inline_start..].find('`') else {
            remaining.push_str(&text[open_idx..]);
            cursor = text.len();
            break;
        };
        let inline_end = inline_start + close_rel;
        let block = &text[inline_start..inline_end];
        if let Some(call) = parse_qwen_inline_tool_invocation(block, calls.len()) {
            calls.push(call);
        } else {
            remaining.push('`');
            remaining.push_str(block);
            remaining.push('`');
        }
        cursor = inline_end + 1;
    }

    if cursor < text.len() {
        remaining.push_str(&text[cursor..]);
    }

    QwenParsedToolCalls {
        remaining_text: remaining.trim().to_string(),
        calls,
    }
}

fn parse_qwen_fenced_bash_call(block: &str, index: usize) -> Option<ResponseToolCall> {
    let trimmed = block.trim();
    let mut lines = trimmed.lines();
    let language = lines.next().unwrap_or_default().trim().to_ascii_lowercase();
    if !matches!(language.as_str(), "bash" | "sh" | "shell" | "zsh") {
        return None;
    }
    let command = lines.collect::<Vec<_>>().join("\n").trim().to_string();
    if command.is_empty() {
        return None;
    }
    Some(qwen_tool_call(
        index,
        "bash",
        serde_json::Map::from_iter([("command".to_string(), Value::String(command))]),
    ))
}

fn parse_qwen_inline_tool_invocation(block: &str, index: usize) -> Option<ResponseToolCall> {
    let normalized = normalize_quotes(block);
    let trimmed = normalized.trim();
    let open_paren = trimmed.find('(')?;
    let close_paren = trimmed.rfind(')')?;
    if close_paren <= open_paren || !trimmed[close_paren + 1..].trim().is_empty() {
        return None;
    }

    let tool_name = trimmed[..open_paren].trim();
    let normalized_name = normalize_qwen_tool_name(tool_name);
    if !is_qwen_tool_name_candidate(&normalized_name) {
        return None;
    }

    let raw_args = trimmed[open_paren + 1..close_paren].trim();
    if raw_args.is_empty() {
        return None;
    }

    let input = match serde_json::from_str::<Value>(raw_args).ok() {
        Some(Value::Object(object)) => object,
        _ => {
            let parsed = parse_qwen_kv_pairs(raw_args);
            if !parsed.is_empty() {
                parsed
            } else if let Some(positional) = parse_qwen_positional_args(raw_args, &normalized_name)
            {
                positional
            } else {
                return None;
            }
        }
    };

    Some(qwen_tool_call(index, &normalized_name, input))
}

fn parse_qwen_positional_args(
    raw_args: &str,
    tool_name: &str,
) -> Option<serde_json::Map<String, Value>> {
    let values = serde_json::from_str::<Vec<Value>>(&format!("[{raw_args}]")).ok()?;
    match tool_name {
        "edit_file" if values.len() >= 3 => {
            let mut input = serde_json::Map::new();
            input.insert("path".to_string(), values.first()?.clone());
            input.insert("old_string".to_string(), values.get(1)?.clone());
            input.insert("new_string".to_string(), values.get(2)?.clone());
            if let Some(replace_all) = values.get(3) {
                input.insert("replace_all".to_string(), replace_all.clone());
            }
            Some(input)
        }
        _ => None,
    }
}

fn parse_qwen_tool_name_and_args_tag_block(block: &str, index: usize) -> Option<ResponseToolCall> {
    let tool_name = qwen_tag_values(block, "tool_name")
        .first()
        .copied()
        .or_else(|| {
            qwen_tag_values_with_close(block, "tool_name", "tool")
                .first()
                .copied()
        })?
        .trim();
    if tool_name.is_empty() {
        return None;
    }
    let normalized_name = normalize_qwen_tool_name(tool_name);
    if !is_qwen_tool_name_candidate(&normalized_name) {
        return None;
    }

    let raw_args = qwen_tag_values(block, "tool_args")
        .first()
        .map(|value| normalize_quotes(value))
        .unwrap_or_default();
    if raw_args.trim().is_empty() {
        return None;
    }
    let input = match serde_json::from_str::<Value>(&raw_args).ok() {
        Some(Value::Object(object)) => object,
        _ => {
            let parsed = parse_qwen_kv_pairs(&raw_args);
            if parsed.is_empty() {
                return None;
            }
            parsed
        }
    };

    Some(qwen_tool_call(index, &normalized_name, input))
}

fn parse_qwen_flat_tool_call(block: &str, index: usize) -> Option<ResponseToolCall> {
    let normalized = normalize_quotes(block);
    let (tool_name, raw_args) = split_tool_name_and_args(&normalized)?;
    let mut input = parse_qwen_kv_pairs(raw_args);
    if input.is_empty() {
        input.insert(
            "raw".to_string(),
            Value::String(raw_args.trim().to_string()),
        );
    }
    Some(qwen_tool_call(index, tool_name, input))
}

fn parse_qwen_named_tool_tag_block(block: &str, index: usize) -> Option<ResponseToolCall> {
    let names = qwen_tag_values(block, "name");
    let tool_name = names.first()?.trim();
    if tool_name.is_empty() {
        return None;
    }

    let mut input = serde_json::Map::new();
    if let Some(arg) = names
        .get(1)
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        let key = match normalize_qwen_tool_name(tool_name).as_str() {
            "read_file" => "path",
            "write_file" => "path",
            "edit_file" => "path",
            "bash" => "command",
            "SemanticSearch" => "query",
            "grep_search" => "pattern",
            "glob_search" => "pattern",
            _ => "input",
        };
        input.insert(key.to_string(), qwen_parameter_value(arg));
    }

    for (idx, value) in names.iter().enumerate().skip(2) {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            continue;
        }
        input.insert(format!("arg{idx}"), qwen_parameter_value(trimmed));
    }

    Some(qwen_tool_call(index, tool_name, input))
}

fn qwen_tag_values<'a>(input: &'a str, tag: &str) -> Vec<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    qwen_tag_values_with_tokens(input, &open, &close)
}

fn qwen_tag_values_with_close<'a>(input: &'a str, open_tag: &str, close_tag: &str) -> Vec<&'a str> {
    let open = format!("<{open_tag}>");
    let close = format!("</{close_tag}>");
    qwen_tag_values_with_tokens(input, &open, &close)
}

fn qwen_tag_values_with_tokens<'a>(input: &'a str, open: &str, close: &str) -> Vec<&'a str> {
    let mut values = Vec::new();
    let mut cursor = 0usize;
    while let Some(open_rel) = input[cursor..].find(&open) {
        let value_start = cursor + open_rel + open.len();
        let Some(close_rel) = input[value_start..].find(&close) else {
            break;
        };
        let value_end = value_start + close_rel;
        values.push(&input[value_start..value_end]);
        cursor = value_end + close.len();
    }
    values
}

fn split_tool_name_and_args(block: &str) -> Option<(&str, &str)> {
    let trimmed = block.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(idx) = first_param_assignment_index(trimmed) {
        let name = trimmed[..idx].trim();
        let args = trimmed[idx..].trim();
        if !name.is_empty() && !args.is_empty() {
            return Some((name, args));
        }
    }
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let name = parts.next()?.trim();
    let args = parts.next()?.trim();
    if name.is_empty() || args.is_empty() {
        return None;
    }
    Some((name, args))
}

fn first_param_assignment_index(input: &str) -> Option<usize> {
    const COMMON_KEYS: &[&str] = &[
        "query", "path", "pattern", "command", "file", "symbol", "line", "column", "content",
        "text", "name", "tool",
    ];
    let lowered = input.to_ascii_lowercase();
    COMMON_KEYS
        .iter()
        .filter_map(|key| lowered.find(&format!("{key}=")))
        .min()
}

fn parse_qwen_kv_pairs(input: &str) -> serde_json::Map<String, Value> {
    let mut result = serde_json::Map::new();
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;

    while i < len {
        while i < len && (bytes[i].is_ascii_whitespace() || bytes[i] == b',') {
            i += 1;
        }
        if i >= len {
            break;
        }

        let key_start = i;
        while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'-')
        {
            i += 1;
        }
        if i == key_start {
            break;
        }
        let key = &input[key_start..i];

        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= len || bytes[i] != b'=' {
            break;
        }
        i += 1;
        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }

        let value = if i < len {
            let quote = bytes[i];
            if matches!(quote, b'"' | b'\'') {
                i += 1;
                let value_start = i;
                while i < len && bytes[i] != quote {
                    i += 1;
                }
                let raw = &input[value_start..i.min(len)];
                if i < len {
                    i += 1;
                }
                raw
            } else {
                let value_start = i;
                while i < len && !bytes[i].is_ascii_whitespace() && bytes[i] != b',' {
                    i += 1;
                }
                &input[value_start..i]
            }
        } else {
            ""
        };

        result.insert(key.to_string(), qwen_parameter_value(value));
    }

    result
}

fn qwen_tool_call(
    index: usize,
    tool_name: &str,
    input: serde_json::Map<String, Value>,
) -> ResponseToolCall {
    let normalized_name = normalize_qwen_tool_name(tool_name);
    let normalized_input = normalize_qwen_tool_input(&normalized_name, input);
    ResponseToolCall {
        id: format!("qwen_tool_call_{index}"),
        function: ResponseToolFunction {
            name: normalized_name,
            arguments: Value::Object(normalized_input).to_string(),
        },
    }
}

fn is_qwen_tool_name_candidate(name: &str) -> bool {
    matches!(
        name,
        "bash"
            | "read_file"
            | "write_file"
            | "edit_file"
            | "glob_search"
            | "grep_search"
            | "SemanticSearch"
            | "ToolSearch"
            | "LSP"
            | "MCPTool"
            | "ListMcpResourcesTool"
            | "ReadMcpResourceTool"
    ) || name.starts_with("mcp__")
}

fn normalize_qwen_tool_name(name: &str) -> String {
    let trimmed = name.trim();
    let token = trimmed
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_' || *ch == '-' || *ch == '/')
        .collect::<String>();

    if let Some(canonical) = canonicalize_qwen_tool_name(&token) {
        return canonical.to_string();
    }

    for prefix in [
        "tool_call",
        "toolcall",
        "tool_use",
        "tooluse",
        "tool",
        "python3",
        "python",
        "py",
    ] {
        if let Some(stripped) = token.strip_prefix(prefix) {
            let candidate = stripped.trim_start_matches(|ch: char| ch == '_' || ch == '-');
            if let Some(canonical) = canonicalize_qwen_tool_name(candidate) {
                return canonical.to_string();
            }
            if candidate.starts_with("mcp__") {
                return candidate.to_string();
            }
        }
    }

    token.to_string()
}

fn canonicalize_qwen_tool_name(name: &str) -> Option<&'static str> {
    match name.to_ascii_lowercase().as_str() {
        "bash" => Some("bash"),
        "read_file" | "readfile" => Some("read_file"),
        "write_file" | "writefile" => Some("write_file"),
        "edit_file" | "editfile" => Some("edit_file"),
        "glob_search" | "globsearch" => Some("glob_search"),
        "grep_search" | "grepsearch" => Some("grep_search"),
        "semanticsearch" | "semantic_search" => Some("SemanticSearch"),
        "toolsearch" | "tool_search" => Some("ToolSearch"),
        "lsp" => Some("LSP"),
        "mcptool" => Some("MCPTool"),
        "listmcpresourcestool" => Some("ListMcpResourcesTool"),
        "readmcpresourcetool" => Some("ReadMcpResourceTool"),
        _ => None,
    }
}

fn normalize_qwen_tool_input(
    tool_name: &str,
    mut input: serde_json::Map<String, Value>,
) -> serde_json::Map<String, Value> {
    match tool_name {
        "read_file" | "write_file" | "edit_file" => {
            if !input.contains_key("path") {
                for alias in ["file", "file_path", "filepath"] {
                    if let Some(value) = input.remove(alias) {
                        input.insert("path".to_string(), value);
                        break;
                    }
                }
            }
            if tool_name == "edit_file" {
                if !input.contains_key("old_string") {
                    for alias in ["search", "old", "find"] {
                        if let Some(value) = input.remove(alias) {
                            input.insert("old_string".to_string(), value);
                            break;
                        }
                    }
                }
                if !input.contains_key("new_string") {
                    for alias in ["replace", "new", "replacement"] {
                        if let Some(value) = input.remove(alias) {
                            input.insert("new_string".to_string(), value);
                            break;
                        }
                    }
                }
            }
        }
        "grep_search" => {
            if !input.contains_key("pattern") {
                for alias in ["query", "text", "input"] {
                    if let Some(value) = input.remove(alias) {
                        input.insert("pattern".to_string(), value);
                        break;
                    }
                }
            }
            if !input.contains_key("path") {
                if let Some(value) = input.remove("file") {
                    input.insert("path".to_string(), value);
                }
            }
        }
        "SemanticSearch" => {
            if !input.contains_key("query") {
                for alias in ["pattern", "text", "input"] {
                    if let Some(value) = input.remove(alias) {
                        input.insert("query".to_string(), value);
                        break;
                    }
                }
            }
        }
        "bash" => {
            if !input.contains_key("command") {
                for alias in ["cmd", "input", "text"] {
                    if let Some(value) = input.remove(alias) {
                        input.insert("command".to_string(), value);
                        break;
                    }
                }
            }
        }
        _ => {}
    }
    input
}

fn normalize_quotes(input: &str) -> String {
    input.replace(['“', '”'], "\"").replace(['‘', '’'], "'")
}

fn parse_qwen_textual_tool_call_block(block: &str, index: usize) -> Option<ResponseToolCall> {
    const FUNCTION_PREFIX: &str = "<function=";
    const FUNCTION_CLOSE: &str = "</function>";
    const PARAMETER_PREFIX: &str = "<parameter=";
    const PARAMETER_CLOSE: &str = "</parameter>";

    let function_start = block.find(FUNCTION_PREFIX)?;
    let function_name_start = function_start + FUNCTION_PREFIX.len();
    let function_name_end_rel = block[function_name_start..].find('>')?;
    let function_name_end = function_name_start + function_name_end_rel;
    let function_name_raw = block[function_name_start..function_name_end].trim();
    let function_name = function_name_raw
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string();
    if function_name.is_empty() {
        return None;
    }

    let body_start = function_name_end + 1;
    let body_end = block[body_start..]
        .find(FUNCTION_CLOSE)
        .map_or(block.len(), |pos| body_start + pos);
    let body = &block[body_start..body_end];

    let mut input = serde_json::Map::new();
    let mut cursor = 0usize;
    while let Some(param_rel) = body[cursor..].find(PARAMETER_PREFIX) {
        let param_start = cursor + param_rel;
        let key_start = param_start + PARAMETER_PREFIX.len();
        let Some(key_end_rel) = body[key_start..].find('>') else {
            break;
        };
        let key_end = key_start + key_end_rel;
        let key = body[key_start..key_end]
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
        if key.is_empty() {
            break;
        }
        let value_start = key_end + 1;
        let Some(value_end_rel) = body[value_start..].find(PARAMETER_CLOSE) else {
            break;
        };
        let value_end = value_start + value_end_rel;
        let raw_value = &body[value_start..value_end];
        input.insert(key, qwen_parameter_value(raw_value));
        cursor = value_end + PARAMETER_CLOSE.len();
    }

    if input.is_empty() {
        return None;
    }

    Some(ResponseToolCall {
        id: format!("qwen_tool_call_{index}"),
        function: ResponseToolFunction {
            name: function_name,
            arguments: Value::Object(input).to_string(),
        },
    })
}

fn qwen_parameter_value(raw: &str) -> Value {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Value::String(String::new());
    }
    serde_json::from_str(trimmed).unwrap_or_else(|_| Value::String(trimmed.to_string()))
}

fn next_sse_frame(buffer: &mut Vec<u8>) -> Option<String> {
    let separator = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|position| (position, 2))
        .or_else(|| {
            buffer
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|position| (position, 4))
        })?;

    let (position, separator_len) = separator;
    let frame = buffer.drain(..position + separator_len).collect::<Vec<_>>();
    let frame_len = frame.len().saturating_sub(separator_len);
    Some(String::from_utf8_lossy(&frame[..frame_len]).into_owned())
}

fn parse_sse_frame(
    frame: &str,
    provider: &str,
    model: &str,
) -> Result<Option<ChatCompletionChunk>, ApiError> {
    let trimmed = frame.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let mut data_lines = Vec::new();
    for line in trimmed.lines() {
        if line.starts_with(':') {
            continue;
        }
        if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start());
        }
    }
    if data_lines.is_empty() {
        return Ok(None);
    }
    let payload = data_lines.join("\n");
    if payload == "[DONE]" {
        return Ok(None);
    }
    serde_json::from_str::<ChatCompletionChunk>(&payload)
        .map(Some)
        .map_err(|error| ApiError::json_deserialize(provider, model, &payload, error))
}

fn read_env_non_empty(key: &str) -> Result<Option<String>, ApiError> {
    match std::env::var(key) {
        Ok(value) if !value.is_empty() => Ok(Some(value)),
        Ok(_) | Err(std::env::VarError::NotPresent) => Ok(super::dotenv_value(key)),
        Err(error) => Err(ApiError::from(error)),
    }
}

#[must_use]
pub fn has_api_key(key: &str) -> bool {
    read_env_non_empty(key)
        .ok()
        .and_then(std::convert::identity)
        .is_some()
}

#[must_use]
pub fn read_base_url(config: OpenAiCompatConfig) -> String {
    if config.provider_name == "OpenAI" {
        if let Ok(value) = std::env::var("LLM_BASE_URL") {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return normalize_openai_base_url(trimmed);
            }
        }
    }
    let configured = std::env::var(config.base_url_env)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let resolved = configured.unwrap_or_else(|| config.default_base_url.to_string());
    if config.provider_name == "OpenAI" {
        normalize_openai_base_url(&resolved)
    } else {
        resolved
    }
}

fn chat_completions_endpoint(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/chat/completions")
    }
}

fn is_local_base_url(base_url: &str) -> bool {
    let lower = base_url.to_ascii_lowercase();
    lower.starts_with("http://127.0.0.1")
        || lower.starts_with("https://127.0.0.1")
        || lower.starts_with("http://localhost")
        || lower.starts_with("https://localhost")
        || lower.starts_with("http://0.0.0.0")
        || lower.starts_with("https://0.0.0.0")
        || lower.starts_with("http://[::1]")
        || lower.starts_with("https://[::1]")
}

fn normalize_openai_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return DEFAULT_OPENAI_BASE_URL.to_string();
    }
    if trimmed.ends_with("/v1") || trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1")
    }
}

fn request_id_from_headers(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get(REQUEST_ID_HEADER)
        .or_else(|| headers.get(ALT_REQUEST_ID_HEADER))
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
}

async fn expect_success(response: reqwest::Response) -> Result<reqwest::Response, ApiError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let request_id = request_id_from_headers(response.headers());
    let body = response.text().await.unwrap_or_default();
    let parsed_error = serde_json::from_str::<ErrorEnvelope>(&body).ok();
    let retryable = is_retryable_status(status);

    Err(ApiError::Api {
        status,
        error_type: parsed_error
            .as_ref()
            .and_then(|error| error.error.error_type.clone()),
        message: parsed_error
            .as_ref()
            .and_then(|error| error.error.message.clone()),
        request_id,
        body,
        retryable,
    })
}

const fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 409 | 429 | 500 | 502 | 503 | 504)
}

fn normalize_finish_reason(value: &str) -> String {
    match value {
        "stop" => "end_turn",
        "tool_calls" => "tool_use",
        other => other,
    }
    .to_string()
}

trait StringExt {
    fn if_empty_then(self, fallback: String) -> String;
}

impl StringExt for String {
    fn if_empty_then(self, fallback: String) -> String {
        if self.is_empty() {
            fallback
        } else {
            self
        }
    }
}

fn first_non_empty_text_trimmed(
    candidates: impl IntoIterator<Item = Option<String>>,
) -> Option<String> {
    candidates
        .into_iter()
        .flatten()
        .map(|value| value.trim().to_string())
        .find(|value| !value.is_empty())
}

fn first_non_empty_text_preserving_whitespace(
    candidates: impl IntoIterator<Item = Option<String>>,
) -> Option<String> {
    candidates
        .into_iter()
        .flatten()
        .find(|value| !value.trim().is_empty())
}

fn reasoning_value_to_text(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(text)) if !text.trim().is_empty() => Some(text.trim().to_string()),
        Some(Value::Array(items)) => {
            let joined = items
                .iter()
                .filter_map(|item| match item {
                    Value::String(text) => Some(text.trim().to_string()),
                    Value::Object(object) => object
                        .get("text")
                        .and_then(Value::as_str)
                        .or_else(|| object.get("content").and_then(Value::as_str))
                        .map(str::trim)
                        .map(ToString::to_string),
                    _ => None,
                })
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            if joined.is_empty() {
                None
            } else {
                Some(joined)
            }
        }
        _ => None,
    }
}

fn message_text_content(message: &ChatMessage) -> Option<String> {
    first_non_empty_text_trimmed([
        message.content.clone(),
        message.reasoning_content.clone(),
        reasoning_value_to_text(message.reasoning.as_ref()),
    ])
}

fn delta_text_content(delta: &ChunkDelta) -> Option<String> {
    first_non_empty_text_preserving_whitespace([
        delta.content.clone(),
        delta.reasoning_content.clone(),
        reasoning_value_to_text(delta.reasoning.as_ref()),
    ])
}

#[cfg(test)]
mod tests {
    use super::{
        build_chat_completion_request, chat_completions_endpoint, is_reasoning_model,
        normalize_finish_reason, openai_tool_choice, parse_tool_arguments, OpenAiCompatClient,
        OpenAiCompatConfig,
    };
    use crate::error::ApiError;
    use crate::types::{
        ContentBlockDelta, ContentBlockStartEvent, InputContentBlock, InputMessage, MessageRequest,
        OutputContentBlock, StreamEvent, ToolChoice, ToolDefinition, ToolResultContentBlock,
    };
    use serde_json::json;
    use std::sync::{Mutex, OnceLock};

    #[test]
    fn request_translation_uses_openai_compatible_shape() {
        let payload = build_chat_completion_request(
            &MessageRequest {
                model: "grok-3".to_string(),
                max_tokens: 64,
                messages: vec![InputMessage {
                    role: "user".to_string(),
                    content: vec![
                        InputContentBlock::Text {
                            text: "hello".to_string(),
                        },
                        InputContentBlock::ToolResult {
                            tool_use_id: "tool_1".to_string(),
                            content: vec![ToolResultContentBlock::Json {
                                value: json!({"ok": true}),
                            }],
                            is_error: false,
                        },
                    ],
                }],
                system: Some("be helpful".to_string()),
                tools: Some(vec![ToolDefinition {
                    name: "weather".to_string(),
                    description: Some("Get weather".to_string()),
                    input_schema: json!({"type": "object"}),
                }]),
                tool_choice: Some(ToolChoice::Auto),
                stream: false,
                ..Default::default()
            },
            OpenAiCompatConfig::xai(),
        );

        assert_eq!(payload["messages"][0]["role"], json!("system"));
        assert_eq!(payload["messages"][1]["role"], json!("user"));
        assert_eq!(payload["messages"][2]["role"], json!("tool"));
        assert_eq!(payload["tools"][0]["type"], json!("function"));
        assert_eq!(payload["tool_choice"], json!("auto"));
    }

    #[test]
    fn tool_schema_object_gets_strict_fields_for_responses_endpoint() {
        // OpenAI /responses endpoint rejects object schemas missing
        // "properties" and "additionalProperties". Verify normalize_object_schema
        // fills them in so the request shape is strict-validator-safe.
        use super::normalize_object_schema;

        // Bare object — no properties at all
        let mut schema = json!({"type": "object"});
        normalize_object_schema(&mut schema);
        assert_eq!(schema["properties"], json!({}));
        assert_eq!(schema["additionalProperties"], json!(false));

        // Nested object inside properties
        let mut schema2 = json!({
            "type": "object",
            "properties": {
                "location": {"type": "object", "properties": {"lat": {"type": "number"}}}
            }
        });
        normalize_object_schema(&mut schema2);
        assert_eq!(schema2["additionalProperties"], json!(false));
        assert_eq!(
            schema2["properties"]["location"]["additionalProperties"],
            json!(false)
        );

        // Existing properties/additionalProperties should not be overwritten
        let mut schema3 = json!({
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": true
        });
        normalize_object_schema(&mut schema3);
        assert_eq!(
            schema3["additionalProperties"],
            json!(true),
            "must not overwrite existing"
        );
    }

    #[test]
    fn reasoning_effort_is_included_when_set() {
        let payload = build_chat_completion_request(
            &MessageRequest {
                model: "o4-mini".to_string(),
                max_tokens: 1024,
                messages: vec![InputMessage::user_text("think hard")],
                reasoning_effort: Some("high".to_string()),
                ..Default::default()
            },
            OpenAiCompatConfig::openai(),
        );
        assert_eq!(payload["reasoning_effort"], json!("high"));
    }

    #[test]
    fn reasoning_effort_omitted_when_not_set() {
        let payload = build_chat_completion_request(
            &MessageRequest {
                model: "gpt-4o".to_string(),
                max_tokens: 64,
                messages: vec![InputMessage::user_text("hello")],
                ..Default::default()
            },
            OpenAiCompatConfig::openai(),
        );
        assert!(payload.get("reasoning_effort").is_none());
    }

    #[test]
    fn openai_streaming_requests_include_usage_opt_in() {
        let payload = build_chat_completion_request(
            &MessageRequest {
                model: "gpt-5".to_string(),
                max_tokens: 64,
                messages: vec![InputMessage::user_text("hello")],
                system: None,
                tools: None,
                tool_choice: None,
                stream: true,
                ..Default::default()
            },
            OpenAiCompatConfig::openai(),
        );

        assert_eq!(payload["stream_options"], json!({"include_usage": true}));
    }

    #[test]
    fn xai_streaming_requests_skip_openai_specific_usage_opt_in() {
        let payload = build_chat_completion_request(
            &MessageRequest {
                model: "grok-3".to_string(),
                max_tokens: 64,
                messages: vec![InputMessage::user_text("hello")],
                system: None,
                tools: None,
                tool_choice: None,
                stream: true,
                ..Default::default()
            },
            OpenAiCompatConfig::xai(),
        );

        assert!(payload.get("stream_options").is_none());
    }

    #[test]
    fn tool_choice_translation_supports_required_function() {
        assert_eq!(openai_tool_choice(&ToolChoice::Any), json!("required"));
        assert_eq!(
            openai_tool_choice(&ToolChoice::Tool {
                name: "weather".to_string(),
            }),
            json!({"type": "function", "function": {"name": "weather"}})
        );
    }

    #[test]
    fn parses_tool_arguments_fallback() {
        assert_eq!(
            parse_tool_arguments("{\"city\":\"Paris\"}"),
            json!({"city": "Paris"})
        );
        assert_eq!(parse_tool_arguments("not-json"), json!({"raw": "not-json"}));
    }

    #[test]
    fn parses_qwen_textual_tool_call_blocks() {
        let payload = "I will edit.\n<tool_call><function=edit_file><parameter=path>/tmp/a.py</parameter><parameter=content>print(1)</parameter></function></tool_call>\nDone";
        let parsed = super::parse_qwen_textual_tool_calls(payload);
        assert_eq!(parsed.calls.len(), 1, "one tool call should be inferred");
        assert_eq!(parsed.remaining_text, "I will edit.\n\nDone");
        assert_eq!(parsed.calls[0].function.name, "edit_file");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&parsed.calls[0].function.arguments)
                .expect("arguments should be valid json"),
            json!({"path":"/tmp/a.py","content":"print(1)"})
        );
    }

    #[test]
    fn parses_qwen_loose_tool_call_markup() {
        let payload =
            "<tool_call>SemanticSearch query=\"API_MAX_POINTS\"</SemanticSearch>\ncontinue";
        let parsed = super::parse_qwen_textual_tool_calls(payload);
        assert_eq!(
            parsed.calls.len(),
            1,
            "one loose tool call should be inferred"
        );
        assert_eq!(parsed.remaining_text, "continue");
        assert_eq!(parsed.calls[0].function.name, "SemanticSearch");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&parsed.calls[0].function.arguments)
                .expect("arguments should be valid json"),
            json!({"query":"API_MAX_POINTS"})
        );
    }

    #[test]
    fn normalizes_qwen_grep_search_file_and_query_aliases() {
        let payload =
            "<tool_call>grep_search file=\"app/config.py\" query=\"API_MAX_POINTS\"</grep_search>";
        let parsed = super::parse_qwen_textual_tool_calls(payload);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].function.name, "grep_search");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&parsed.calls[0].function.arguments)
                .expect("arguments should be valid json"),
            json!({"path":"app/config.py","pattern":"API_MAX_POINTS"})
        );
    }

    #[test]
    fn normalizes_qwen_tool_prefixed_name() {
        let payload =
            "<tool_call>toolSemanticSearch query=\"app/api.py\"</SemanticSearch>\ncontinue";
        let parsed = super::parse_qwen_textual_tool_calls(payload);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].function.name, "SemanticSearch");
    }

    #[test]
    fn normalizes_qwen_toolcall_prefixed_name() {
        let payload = "<tool_call>toolcallgrep_search query=\"API_MAX_POINTS\"</grep_search>";
        let parsed = super::parse_qwen_textual_tool_calls(payload);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].function.name, "grep_search");
    }

    #[test]
    fn parses_qwen_named_tool_tags() {
        let payload =
            "<tool_call><tool><name>read_file</name><name>app/config.py</name></tool></tool_call>";
        let parsed = super::parse_qwen_textual_tool_calls(payload);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].function.name, "read_file");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&parsed.calls[0].function.arguments)
                .expect("arguments should be valid json"),
            json!({"path":"app/config.py"})
        );
    }

    #[test]
    fn parses_qwen_tool_name_and_tool_args_tags() {
        let payload = "<tool_call>tool_use><tool_name>read_file</tool><tool_args>{“path”:“app/config.py”}</tool_args></tool_use>";
        let parsed = super::parse_qwen_textual_tool_calls(payload);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].function.name, "read_file");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&parsed.calls[0].function.arguments)
                .expect("arguments should be valid json"),
            json!({"path":"app/config.py"})
        );
    }

    #[test]
    fn parses_qwen_inline_backtick_tool_invocation() {
        let payload = "I will edit now: `edit_file(file_path=\"/tmp/probe_edit.txt\",search=\"alpha\",replace=\"beta\")`";
        let parsed = super::parse_qwen_textual_tool_calls(payload);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].function.name, "edit_file");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&parsed.calls[0].function.arguments)
                .expect("arguments should be valid json"),
            json!({
                "path":"/tmp/probe_edit.txt",
                "old_string":"alpha",
                "new_string":"beta"
            })
        );
    }

    #[test]
    fn parses_qwen_inline_python_prefixed_positional_edit_call() {
        let payload = "Do it: `pythonedit_file(\"probe_edit.txt\",\"alpha\",\"beta\")`";
        let parsed = super::parse_qwen_textual_tool_calls(payload);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].function.name, "edit_file");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&parsed.calls[0].function.arguments)
                .expect("arguments should be valid json"),
            json!({
                "path":"probe_edit.txt",
                "old_string":"alpha",
                "new_string":"beta"
            })
        );
    }

    #[test]
    fn does_not_promote_unknown_backtick_function_invocation() {
        let payload = "Planning: `not_a_tool(path=\"app/config.py\")`";
        let parsed = super::parse_qwen_textual_tool_calls(payload);
        assert_eq!(parsed.calls.len(), 0, "unknown call should remain text");
        assert_eq!(parsed.remaining_text, payload);
    }

    #[test]
    fn normalizes_qwen_read_file_name_and_file_key_alias() {
        let payload = "<tool_call>read_file( file=app/config.py</tool_call>";
        let parsed = super::parse_qwen_textual_tool_calls(payload);
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].function.name, "read_file");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&parsed.calls[0].function.arguments)
                .expect("arguments should be valid json"),
            json!({"path":"app/config.py"})
        );
    }

    #[test]
    fn does_not_promote_inline_bash_backticks_without_explicit_tool_tag() {
        let payload = "Let me inspect with `bash rg -n \"API_MAX_POINTS\" app/config.py` first.";
        let parsed = super::parse_qwen_textual_tool_calls(payload);
        assert_eq!(
            parsed.calls.len(),
            0,
            "inline command text should remain text"
        );
        assert_eq!(parsed.remaining_text, payload);
    }

    #[test]
    fn normalize_response_promotes_qwen_textual_tool_calls_when_native_calls_missing() {
        let response: super::ChatCompletionResponse = serde_json::from_value(json!({
            "id": "chatcmpl-qwen-text",
            "model": "Qwen_Qwen3.5-4B-Q4_K_M.gguf",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<tool_call><function=glob_search><parameter=path>/repo</parameter><parameter=pattern>**/*.py</parameter></function></tool_call>"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        }))
        .expect("response json should deserialize");

        let normalized =
            super::normalize_response("qwen3.5:4b", response).expect("response should normalize");
        assert_eq!(normalized.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(normalized.content.len(), 1);
        match &normalized.content[0] {
            OutputContentBlock::ToolUse { name, input, .. } => {
                assert_eq!(name, "glob_search");
                assert_eq!(input, &json!({"path":"/repo","pattern":"**/*.py"}));
            }
            other => panic!("expected tool_use block, got {other:?}"),
        }
    }

    #[test]
    fn normalize_response_supports_legacy_function_call_shape() {
        let response: super::ChatCompletionResponse = serde_json::from_value(json!({
            "id": "chatcmpl-func",
            "model": "qwen3.5:4b",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"README.md\"}"
                    }
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        }))
        .expect("response json should deserialize");

        let normalized =
            super::normalize_response("qwen3.5:4b", response).expect("response should normalize");
        assert_eq!(normalized.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(normalized.content.len(), 1);
        match &normalized.content[0] {
            OutputContentBlock::ToolUse { name, input, .. } => {
                assert_eq!(name, "read_file");
                assert_eq!(input, &json!({"path":"README.md"}));
            }
            other => panic!("expected tool_use block, got {other:?}"),
        }
    }

    #[test]
    fn missing_xai_api_key_is_provider_specific() {
        let _lock = env_lock();
        std::env::remove_var("XAI_API_KEY");
        let error = OpenAiCompatClient::from_env(OpenAiCompatConfig::xai())
            .expect_err("missing key should error");
        assert!(matches!(
            error,
            ApiError::MissingCredentials {
                provider: "xAI",
                ..
            }
        ));
    }

    #[test]
    fn endpoint_builder_accepts_base_urls_and_full_endpoints() {
        assert_eq!(
            chat_completions_endpoint("https://api.x.ai/v1"),
            "https://api.x.ai/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_endpoint("https://api.x.ai/v1/"),
            "https://api.x.ai/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_endpoint("https://api.x.ai/v1/chat/completions"),
            "https://api.x.ai/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_endpoint("http://127.0.0.1:8080"),
            "http://127.0.0.1:8080/chat/completions"
        );
    }

    #[test]
    fn default_openai_base_url_is_local_runtime_first() {
        assert_eq!(super::DEFAULT_OPENAI_BASE_URL, "http://127.0.0.1:8080/v1");
    }

    #[test]
    fn openai_base_url_prefers_llm_base_url_alias() {
        let _lock = env_lock();
        std::env::set_var("LLM_BASE_URL", "http://127.0.0.1:8129");
        std::env::remove_var("OPENAI_BASE_URL");
        assert_eq!(
            super::read_base_url(OpenAiCompatConfig::openai()),
            "http://127.0.0.1:8129/v1"
        );
        std::env::remove_var("LLM_BASE_URL");
    }

    #[test]
    fn openai_from_env_allows_missing_key_for_local_runtime() {
        let _lock = env_lock();
        std::env::remove_var("OPENAI_API_KEY");
        std::env::set_var("OPENAI_BASE_URL", "http://127.0.0.1:8080");
        let result = OpenAiCompatClient::from_env(OpenAiCompatConfig::openai());
        assert!(result.is_ok());
        std::env::remove_var("OPENAI_BASE_URL");
    }

    #[test]
    fn openai_from_env_requires_key_for_remote_runtime() {
        let _lock = env_lock();
        std::env::remove_var("OPENAI_API_KEY");
        std::env::set_var("OPENAI_BASE_URL", "https://api.openai.com/v1");
        let error = OpenAiCompatClient::from_env(OpenAiCompatConfig::openai())
            .expect_err("remote openai should require api key");
        assert!(matches!(
            error,
            ApiError::MissingCredentials {
                provider: "OpenAI",
                ..
            }
        ));
        std::env::remove_var("OPENAI_BASE_URL");
    }

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[test]
    fn normalizes_stop_reasons() {
        assert_eq!(normalize_finish_reason("stop"), "end_turn");
        assert_eq!(normalize_finish_reason("tool_calls"), "tool_use");
    }

    #[test]
    fn tuning_params_included_in_payload_when_set() {
        let request = MessageRequest {
            model: "gpt-4o".to_string(),
            max_tokens: 1024,
            messages: vec![],
            stream: false,
            temperature: Some(0.7),
            top_p: Some(0.9),
            frequency_penalty: Some(0.5),
            presence_penalty: Some(0.3),
            stop: Some(vec!["\n".to_string()]),
            ..Default::default()
        };
        let payload = build_chat_completion_request(&request, OpenAiCompatConfig::openai());
        assert_eq!(payload["temperature"], 0.7);
        assert_eq!(payload["top_p"], 0.9);
        assert_eq!(payload["frequency_penalty"], 0.5);
        assert_eq!(payload["presence_penalty"], 0.3);
        assert_eq!(payload["stop"], json!(["\n"]));
    }

    #[test]
    fn local_qwen_payload_includes_runtime_extension_fields() {
        let request = MessageRequest {
            model: "qwen3.5:4b".to_string(),
            max_tokens: 1024,
            messages: vec![],
            stream: false,
            top_k: Some(20),
            min_p: Some(0.0),
            repeat_penalty: Some(1.01),
            repeat_last_n: Some(128),
            chat_template_kwargs: Some(json!({ "enable_thinking": false })),
            ..Default::default()
        };
        let payload = super::build_chat_completion_request_with_base_url(
            &request,
            OpenAiCompatConfig::openai(),
            "http://127.0.0.1:8129/v1",
        );
        assert_eq!(payload["top_k"], 20);
        assert_eq!(payload["min_p"], 0.0);
        assert_eq!(payload["repeat_penalty"], 1.01);
        assert_eq!(payload["repeat_last_n"], 128);
        assert_eq!(
            payload["chat_template_kwargs"],
            json!({ "enable_thinking": false })
        );
    }

    #[test]
    fn remote_openai_payload_strips_runtime_extension_fields() {
        let request = MessageRequest {
            model: "qwen3.5:4b".to_string(),
            max_tokens: 1024,
            messages: vec![],
            stream: false,
            top_k: Some(20),
            min_p: Some(0.0),
            repeat_penalty: Some(1.01),
            repeat_last_n: Some(128),
            chat_template_kwargs: Some(json!({ "enable_thinking": false })),
            ..Default::default()
        };
        let payload = super::build_chat_completion_request_with_base_url(
            &request,
            OpenAiCompatConfig::openai(),
            "https://api.openai.com/v1",
        );
        assert!(
            payload.get("top_k").is_none(),
            "top_k should not be emitted to remote OpenAI endpoints"
        );
        assert!(payload.get("min_p").is_none());
        assert!(payload.get("repeat_penalty").is_none());
        assert!(payload.get("repeat_last_n").is_none());
        assert!(payload.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn reasoning_model_strips_tuning_params() {
        let request = MessageRequest {
            model: "o1-mini".to_string(),
            max_tokens: 1024,
            messages: vec![],
            stream: false,
            temperature: Some(0.7),
            top_p: Some(0.9),
            frequency_penalty: Some(0.5),
            presence_penalty: Some(0.3),
            stop: Some(vec!["\n".to_string()]),
            ..Default::default()
        };
        let payload = build_chat_completion_request(&request, OpenAiCompatConfig::openai());
        assert!(
            payload.get("temperature").is_none(),
            "reasoning model should strip temperature"
        );
        assert!(
            payload.get("top_p").is_none(),
            "reasoning model should strip top_p"
        );
        assert!(payload.get("frequency_penalty").is_none());
        assert!(payload.get("presence_penalty").is_none());
        // stop is safe for all providers
        assert_eq!(payload["stop"], json!(["\n"]));
    }

    #[test]
    fn grok_3_mini_is_reasoning_model() {
        assert!(is_reasoning_model("grok-3-mini"));
        assert!(is_reasoning_model("o1"));
        assert!(is_reasoning_model("o1-mini"));
        assert!(is_reasoning_model("o3-mini"));
        assert!(!is_reasoning_model("gpt-4o"));
        assert!(!is_reasoning_model("grok-3"));
        assert!(!is_reasoning_model("claude-sonnet-4-6"));
    }

    #[test]
    fn qwen_reasoning_variants_are_detected() {
        // QwQ reasoning model
        assert!(is_reasoning_model("qwen-qwq-32b"));
        assert!(is_reasoning_model("qwen/qwen-qwq-32b"));
        // Qwen3 thinking family
        assert!(is_reasoning_model("qwen3-30b-a3b-thinking"));
        assert!(is_reasoning_model("qwen/qwen3-30b-a3b-thinking"));
        // Bare qwq
        assert!(is_reasoning_model("qwq-plus"));
        // Regular Qwen models must NOT be classified as reasoning
        assert!(!is_reasoning_model("qwen-max"));
        assert!(!is_reasoning_model("qwen/qwen-plus"));
        assert!(!is_reasoning_model("qwen-turbo"));
    }

    #[test]
    fn tuning_params_omitted_from_payload_when_none() {
        let request = MessageRequest {
            model: "gpt-4o".to_string(),
            max_tokens: 1024,
            messages: vec![],
            stream: false,
            ..Default::default()
        };
        let payload = build_chat_completion_request(&request, OpenAiCompatConfig::openai());
        assert!(
            payload.get("temperature").is_none(),
            "temperature should be absent"
        );
        assert!(payload.get("top_p").is_none(), "top_p should be absent");
        assert!(payload.get("frequency_penalty").is_none());
        assert!(payload.get("presence_penalty").is_none());
        assert!(payload.get("stop").is_none());
    }

    #[test]
    fn gpt5_uses_max_completion_tokens_not_max_tokens() {
        // gpt-5* models require `max_completion_tokens`; legacy `max_tokens` causes
        // a request-validation failure. Verify the correct key is emitted.
        let request = MessageRequest {
            model: "gpt-5.2".to_string(),
            max_tokens: 512,
            messages: vec![],
            stream: false,
            ..Default::default()
        };
        let payload = build_chat_completion_request(&request, OpenAiCompatConfig::openai());
        assert_eq!(
            payload["max_completion_tokens"],
            json!(512),
            "gpt-5.2 should emit max_completion_tokens"
        );
        assert!(
            payload.get("max_tokens").is_none(),
            "gpt-5.2 must not emit max_tokens"
        );
    }

    #[test]
    fn non_gpt5_uses_max_tokens() {
        // Older OpenAI models expect `max_tokens`; verify gpt-4o is unaffected.
        let request = MessageRequest {
            model: "gpt-4o".to_string(),
            max_tokens: 512,
            messages: vec![],
            stream: false,
            ..Default::default()
        };
        let payload = build_chat_completion_request(&request, OpenAiCompatConfig::openai());
        assert_eq!(payload["max_tokens"], json!(512));
        assert!(
            payload.get("max_completion_tokens").is_none(),
            "gpt-4o must not emit max_completion_tokens"
        );
    }

    #[test]
    fn normalize_response_uses_reasoning_content_when_text_content_is_empty() {
        let response: super::ChatCompletionResponse = serde_json::from_value(json!({
            "id": "chatcmpl-test",
            "model": "Qwen_Qwen3.5-4B-Q4_K_M.gguf",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "FINAL ANSWER: ready"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4
            }
        }))
        .expect("response json should deserialize");

        let normalized =
            super::normalize_response("qwen3.5:4b", response).expect("response should normalize");
        assert_eq!(
            normalized.content.len(),
            1,
            "fallback text should be emitted"
        );
        match &normalized.content[0] {
            OutputContentBlock::Text { text } => assert_eq!(text, "FINAL ANSWER: ready"),
            other => panic!("expected text content block, got {other:?}"),
        }
    }

    #[test]
    fn stream_state_emits_text_delta_from_reasoning_content_when_content_missing() {
        let chunk: super::ChatCompletionChunk = serde_json::from_value(json!({
            "id": "chatcmpl-stream-test",
            "model": "gpt-4o-mini",
            "choices": [{
                "delta": {
                    "reasoning_content": "step-by-step"
                }
            }]
        }))
        .expect("chunk json should deserialize");

        let mut state = super::StreamState::new("gpt-4o-mini".to_string());
        let events = state.ingest_chunk(chunk).expect("chunk should be ingested");
        let reasoning_delta = events.into_iter().any(|event| {
            matches!(
                event,
                StreamEvent::ContentBlockDelta(delta_event)
                    if matches!(
                        delta_event.delta,
                        ContentBlockDelta::TextDelta { ref text } if text == "step-by-step"
                    )
            )
        });
        assert!(
            reasoning_delta,
            "stream should emit text delta from reasoning_content fallback"
        );
    }

    #[test]
    fn stream_state_promotes_qwen_textual_tool_calls_from_text_deltas() {
        let chunk: super::ChatCompletionChunk = serde_json::from_value(json!({
            "id": "chatcmpl-stream-qwen-text",
            "model": "Qwen_Qwen3.5-4B-Q4_K_M.gguf",
            "choices": [{
                "delta": {
                    "content": "<tool_call><function=edit_file><parameter=path>/tmp/x.py</parameter></function></tool_call>"
                },
                "finish_reason": "stop"
            }]
        }))
        .expect("chunk json should deserialize");

        let mut state = super::StreamState::new("qwen3.5:4b".to_string());
        let chunk_events = state.ingest_chunk(chunk).expect("chunk should be ingested");
        assert!(
            chunk_events
                .iter()
                .all(|event| !matches!(event, StreamEvent::ContentBlockDelta(_))),
            "qwen text should be buffered until finish"
        );

        let finish_events = state.finish().expect("finish should succeed");
        let saw_tool_start = finish_events.iter().any(|event| {
            matches!(
                event,
                StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                    content_block: OutputContentBlock::ToolUse { name, .. },
                    ..
                }) if name == "edit_file"
            )
        });
        assert!(saw_tool_start, "finish should emit inferred tool call");
        let saw_tool_stop_reason = finish_events.iter().any(|event| {
            matches!(
                event,
                StreamEvent::MessageDelta(delta)
                    if delta.delta.stop_reason.as_deref() == Some("tool_use")
            )
        });
        assert!(saw_tool_stop_reason, "stop reason should become tool_use");
    }

    #[test]
    fn stream_state_supports_legacy_function_call_deltas() {
        let chunk: super::ChatCompletionChunk = serde_json::from_value(json!({
            "id": "chatcmpl-stream-func",
            "model": "qwen3.5:4b",
            "choices": [{
                "delta": {
                    "function_call": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"README.md\"}"
                    }
                },
                "finish_reason": "tool_calls"
            }]
        }))
        .expect("chunk json should deserialize");

        let mut state = super::StreamState::new("qwen3.5:4b".to_string());
        let events = state.ingest_chunk(chunk).expect("chunk should be ingested");
        let saw_tool_start = events.iter().any(|event| {
            matches!(
                event,
                StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                    content_block: OutputContentBlock::ToolUse { name, .. },
                    ..
                }) if name == "read_file"
            )
        });
        assert!(
            saw_tool_start,
            "legacy function_call should map to tool_use"
        );
    }

    #[test]
    fn stream_state_preserves_leading_whitespace_in_text_deltas() {
        let chunk_one: super::ChatCompletionChunk = serde_json::from_value(json!({
            "id": "chatcmpl-stream-text-1",
            "model": "gpt-4o-mini",
            "choices": [{
                "delta": {
                    "content": "Now"
                }
            }]
        }))
        .expect("chunk json should deserialize");

        let chunk_two: super::ChatCompletionChunk = serde_json::from_value(json!({
            "id": "chatcmpl-stream-text-2",
            "model": "gpt-4o-mini",
            "choices": [{
                "delta": {
                    "content": " Iunderstand"
                }
            }]
        }))
        .expect("chunk json should deserialize");

        let mut state = super::StreamState::new("gpt-4o-mini".to_string());
        let first_events = state
            .ingest_chunk(chunk_one)
            .expect("first chunk should be ingested");
        let saw_first_delta = first_events.iter().any(|event| {
            matches!(
                event,
                StreamEvent::ContentBlockDelta(delta_event)
                    if matches!(
                        delta_event.delta,
                        ContentBlockDelta::TextDelta { ref text } if text == "Now"
                    )
            )
        });
        assert!(saw_first_delta, "first delta should be emitted as-is");

        let second_events = state
            .ingest_chunk(chunk_two)
            .expect("second chunk should be ingested");
        let saw_second_delta_with_space = second_events.iter().any(|event| {
            matches!(
                event,
                StreamEvent::ContentBlockDelta(delta_event)
                    if matches!(
                        delta_event.delta,
                        ContentBlockDelta::TextDelta { ref text } if text == " Iunderstand"
                    )
            )
        });
        assert!(
            saw_second_delta_with_space,
            "second delta should preserve leading space"
        );
    }
}
