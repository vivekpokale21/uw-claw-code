# Qwen3.5 Compatibility Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Qwen3.5-first local runtime behavior reliably convert model output into executable tool calls in Rust, with measurable end-to-end improvement.

**Architecture:** Keep OpenAI-compatible transport unchanged, but add Qwen-aware response normalization in the Rust provider layer. Promote textual `<tool_call>` payloads and legacy `function_call` payloads into canonical tool-use blocks consumed by the existing runtime conversation loop.

**Tech Stack:** Rust (`api` crate provider normalization + stream state), existing runtime loop, cargo tests, external capability matrix harness.

---

## Execution Record (Completed)

### Task 1: Qwen textual tool-call promotion in non-stream responses

**Files:**
- Modify: `rust/crates/api/src/providers/openai_compat.rs`

- [x] Added `parse_qwen_textual_tool_calls` and block parser logic for `<tool_call><function=...><parameter=...>` format.
- [x] Wired parser into `normalize_response` when native `tool_calls` are missing on Qwen-family models.
- [x] Ensured stop reason flips to `tool_use` when inferred tool calls are present.

### Task 2: Qwen textual tool-call promotion in streaming responses

**Files:**
- Modify: `rust/crates/api/src/providers/openai_compat.rs`

- [x] Updated stream state to buffer Qwen text deltas and parse at `finish()`.
- [x] Emitted synthetic `ContentBlockStart/Delta/Stop` tool-use events from inferred tool calls.
- [x] Kept existing non-Qwen behavior unchanged.

### Task 3: Legacy function_call compatibility fallback

**Files:**
- Modify: `rust/crates/api/src/providers/openai_compat.rs`

- [x] Added non-stream fallback for `message.function_call` when `tool_calls` absent.
- [x] Added stream fallback for `delta.function_call` when `tool_calls` absent.

### Task 4: Regression tests for Qwen compatibility path

**Files:**
- Modify: `rust/crates/api/src/providers/openai_compat.rs` (test module)

- [x] Added parser unit test for textual tool-call extraction.
- [x] Added non-stream normalization test for inferred Qwen tool calls.
- [x] Added stream finish test for inferred tool calls from buffered text.
- [x] Added legacy `function_call` tests for non-stream and stream paths.

### Task 5: Re-run capability matrix and record outcomes

**Files:**
- Artifacts: `/tmp/qwen_matrix_postpatch_2026-04-09`

- [x] Ran 8 isolated single-turn tasks (fresh process + fresh worktree per task).
- [x] Confirmed stream-content fatal failures remained at `0/8`.
- [x] Observed strict target-task success increased from `1/8` to `3/8`.
- [x] Captured residual failures: timeout-prone long loops, partial edits, and whitespace-collapsed payload corruption on some write paths.

## Immediate Follow-up Plan (Not Yet Implemented)

### Task 6: Add Qwen payload safety guardrails for space-collapsed write/edit arguments

**Files:**
- Modify: `rust/crates/runtime/src/conversation.rs`
- Modify: `rust/crates/tools/src/lib.rs`

- [ ] Add heuristic rejection/retry for suspiciously collapsed code payloads in high-risk write/edit tool calls.
- [ ] Emit explicit runtime error classification for rejected malformed Qwen tool payloads.

### Task 7: Add Qwen no-progress timeout policy tuned for small-context local models

**Files:**
- Modify: `rust/crates/runtime/src/conversation.rs`
- Modify: `rust/crates/rusty-claude-cli/src/main.rs` (if needed for user-facing diagnostics)

- [ ] Add stricter per-turn no-progress timeout and actionable stop reason for long wandering tool loops.
- [ ] Keep policy model-scoped to Qwen family to avoid collateral behavior changes.
