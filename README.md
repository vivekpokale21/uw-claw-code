# uw-claw-code

Local-first Claw fork focused on harness reliability for small/medium open models (Qwen-family on `llama.cpp`).

This repo prioritizes:

- deterministic tool execution over one-shot model cleverness
- robust recovery loops for unattended/AFK operation
- clear runtime diagnostics when provider/tooling behavior degrades

## Core Direction

This fork is centered on hardening the agent harness layer:

- API/provider routing and OpenAI-compatible transport behavior
- tool-call normalization and dispatch reliability
- retrieval quality and guardrails (`SemanticSearch`, `grep`, `LSP`, repo context)
- loop safety (stall detection, compaction, bounded retries)
- operator controls and inspectability (`doctor`, status/checkpoint controls, health snapshots)

Reference design notes in:

- `docs/claw_local_port/AGENTS.md`
- `docs/claw_local_port/PLAN.md`
- `docs/claw_local_port/STATUS.md`
- `docs/claw_local_port/FUTURE.md`

## Major Hardening Work

### 1) Provider and Runtime Routing Hardening

Result:
- model prefix and provider resolution became explicit and predictable in local multi-provider environments.

### 2) OpenAI-Compatible Stream/Response Resilience (Qwen)

Primary code:

- `rust/crates/api/src/providers/openai_compat.rs`

Result:
- fewer dead-end turns from malformed/missing tool-call structures in local Qwen-compatible outputs.

### 3) Tool Surface and Dispatch Hardening

Primary code:

- `rust/crates/tools/src/lib.rs`
- `rust/crates/rusty-claude-cli/src/main.rs`

Hardening themes:

- strict `allowedTools` canonicalization and unsupported-tool rejection
- stable built-in tool registry for read/edit/search/automation flows
- alias normalization (`read`, `write`, `edit`, `semantic`, etc.) into canonical tool names

Result:
- tighter control over what the model can actually call, with deterministic failures instead of silent drift.

### 4) Retrieval Stack Hardening

Primary code:

- `rust/crates/runtime/src/prompt.rs`
- `rust/crates/runtime/src/conversation.rs`
- `rust/crates/tools/src/semantic_search.rs`
- `rust/crates/semantic_search/*`

What was improved:

- repository map injected into prompt context
- explicit retrieval guidance (`SemanticSearch` first, `grep_search` fallback)
- retrieval-evidence enforcement before path-targeted edits
- `SemanticSearch` tool path backed by the new `semantic_search` crate with graceful lexical fallback when embeddings are unavailable

Result:
- better codebase navigation quality and fewer blind edits.

### 5) LSP Reliability and Health Persistence

Primary code:

- `rust/crates/runtime/src/lsp_client.rs`
- `rust/crates/runtime/src/prompt.rs` (`# LSP context`)

Result:
- repeated LSP failures are visible and bounded via cooldown state instead of causing opaque instability.

### 6) AFK/Long-Run Loop Safeguards

Result:
- long unattended sessions fail faster with useful stop reasons and better context-pressure handling.

### 7) Local Runtime Template/Cache Behavior

Primary path:

- `runtime_templates/qwen35_chat_template_cachefix.jinja`

What was addressed:

- chat template behavior adjusted for better local Qwen cache reuse characteristics and reduced prompt churn.

## Runtime Scripts (Operational)

- `scripts/start_qwen35_llama_server.sh`
  - launches `llama-server` with approach/mode profiles and template wiring
- `scripts/qwen35_profiles.sh`
  - planner/executor/repair sampling profiles
- `scripts/print_qwen35_profile.sh`
  - inspect active profile values

## Repository Layout

- `rust/` - Rust runtime, CLI, API providers, tools, command surface
- `runtime_templates/` - local runtime chat templates
- `scripts/` - local launcher/profile/operator scripts
- `docs/claw_local_port/` - AFK local-first design/plan/status docs
- `USAGE.md` - command and operational usage guide
- `STATUS.md`, `ROADMAP.md`, `PARITY.md` - ongoing migration and hardening context

## Quick Start

Build:

```bash
cd rust
cargo build --workspace
```

Run a local-profile llama server:

```bash
./scripts/start_qwen35_llama_server.sh --approach omnicoder --mode executor
```

Run a prompt with local model defaults:

```bash
./rust/target/debug/claw prompt "inspect this repository and propose a safe refactor plan" --dangerously-skip-permissions
```

## Notes

This README is intentionally focused on harness/tool hardening and local runtime reliability.

For full command docs and configuration surface, use `USAGE.md`.
