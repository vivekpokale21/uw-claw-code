# FUTURE.md

Last updated: 2026-04-22

## Migration Slices (AFK Local-First Port)

### Completed

- Slice 1 (2026-04-09): Rust local runtime defaults and OpenAI-compatible base URL behavior now align with local llama.cpp-first operation (`qwen3.5:4b`, localhost `/v1`, `LLM_BASE_URL` precedence, local no-key allowance).
- Slice 2 (2026-04-09): Rust AFK slash parity now includes `/budget`, `/checkpoint`, `/list-runs`, `/show-run`, and REPL unknown-slash passthrough for absolute-path prompts.
- Slice 3 (2026-04-09): Rust turn-loop reliability now includes deterministic `tool_loop_stalled` / `no_progress_stalled` stop reasons, bounded repetition guards, and pre-turn budget/auto-compaction safeguards.
- Slice 4 (2026-04-09): Rust LSP path now persists and reloads health telemetry in `.port_sessions/lsp_health_state.json`, enforces cooldown-backed retry suppression, and exposes `health`/`status` inspection via the Rust LSP tool surface.
- Slice 5 (2026-04-09): Rust CLI UX now reports richer pre-turn statusline context (`pressure`, utilization %, messages/turns/session), and slash parity now includes `/runs` alias plus numeric `/checkpoint <limit>` shorthand.
- Slice 6 (2026-04-09): Test hardening now covers Rust LSP health persistence path/telemetry updates at tool level, and migration docs now explicitly mark Python as compatibility fallback only for AFK behavior.

### Next

- Post-slice migration hardening (history-based reliability tuning + remaining Rust parity gaps).

## Priority Backlog

## P0 — AFK Reliability Core

0. Capability benchmark ladder + retrieval A/B for local Qwen
- Run a progressive edit/test capability ladder on external dataset repos (not product repos) to measure real coding reliability beyond simple single-file tasks.
- Keep runtime fixed (`qwen3.5:4b`, local OpenAI-compatible endpoint) and compare:
  - baseline (no embedding retrieval)
  - retrieval-enabled path (embedding index, planned with `nomic-embed-text` class model)
- Track failures by class:
  - stream/content failures
  - malformed output text formatting
  - incorrect/over-scoped edits
  - verification failures on touched files
- Acceptance criteria:
  - zero stream-content fatal failures in ladder run
  - >= 75% task success at target difficulty
  - measurable improvement in multi-file task success in retrieval-enabled rerun
- Current state (2026-04-09 baseline run):
  - stream-content fatal failures met target (`0/8`)
  - baseline task-success target missed (`1/8` successful target edits in task worktrees)
  - post-compat task-success still below target (`3/8` strict successful target edits in task worktrees)
  - retrieval A/B blocked until baseline no-op/path-targeting failures are reduced
- Current state update (2026-04-22 harness-integrity pass, rerun completed):
  - latest wired fallback reference run (`/tmp/qwen_matrix_semantic_wired_fallback_20260416_122959`) exposed two integrity artifacts:
    - `.semantic_search/` writes were counted as meaningful repo edits
    - already-satisfied tasks still consumed model turns and failed meaningful-change gates
  - ladder harness now includes:
    - artifact-aware meaningful-change filtering (`.claw/`, `.port_sessions/`, `.semantic_search/`)
    - precheck short-circuit (`QWEN35_MATRIX_PRECHECK_TASKS`, `QWEN35_MATRIX_SKIP_IF_PRECHECK_PASS`)
    - phased executor passes for long tasks (`QWEN35_MATRIX_EXECUTOR_PASSES`)
    - stage-specific executor timeout budget (`QWEN35_MATRIX_EXECUTOR_TIMEOUT_SECS`, default `1200s`)
    - executor-only loop stall controls (`QWEN35_MATRIX_EXECUTOR_TOOL_LOOP_STALL_LIMIT`, `QWEN35_MATRIX_EXECUTOR_NO_PROGRESS_STALL_LIMIT`)
    - context-window short-circuit acceptance when objective checks already pass
  - rerun evidence (`/tmp/qwen_matrix_perf_guard_20260422_141026`):
    - strict result: `6/8` task pass, total `1,714,174 ms`, avg `214,271 ms/task`
    - first-7-task wall-time improvement vs prior post-timeout run: `-17.60%` (`1,427,123 ms` vs `1,731,993 ms`)
    - remaining strict-fail cluster stayed concentrated in late tasks (time-filters/client-config docs segment)
  - profile-split follow-up (2026-04-22):
    - harness now supports explicit runtime profiles (`quality`, `balanced`, `fast`) via `QWEN35_MATRIX_RUNTIME_PROFILE`.
    - `balanced` top-3 rerun (`/tmp/qwen_matrix_balanced_top3_20260422_153116`) preserved strict pass (`3/3`) but was slower than prior top-3 reference (`+44.50%` wall time vs `/tmp/qwen_matrix_perf_guard_20260422_141026`).
    - `fast` top-3 rerun (`/tmp/qwen_matrix_fast_top3_20260422_154504`) recovered throughput (`-64.09%` wall time vs same reference) but failed strict integrity (`0/3`) due zero-edit executor exits.
    - post-regression rerun after executor path/toolset hardening (`/tmp/qwen_matrix_post_bash_regression_debug`) kept strict pass (`3/3`) and cut balanced top-3 wall time by `-20.12%` vs prior balanced baseline (`606,924 ms` vs `759,752 ms`).
    - remaining inflation is localized: task 1 `-25.92%`, task 2 `-34.06%`, task 3 `+3.57%` (extra pass/retry churn).
    - follow-up rerun after `no_progress_stalled` objective-success acceptance (`/tmp/qwen_matrix_post_stall_accept_20260422_194524`) kept strict pass (`3/3`) and reduced balanced top-3 wall time further to `499,062 ms` (`-34.31%` vs balanced baseline, `-17.77%` vs post-bash rerun).
    - task-level shift in latest rerun: task 2 `180,100 ms`, task 3 `174,831 ms`; these were the prior churn hotspots.
    - stage metrics now include explicit `bash_calls` for executor drift attribution.
  - experimental objective-gated follow-up (forcing pass/retry when interim checks fail) was tested and reverted after increasing runtime on failing paths without recovering strict-pass on the persistent docs task.
- Immediate follow-up focus:
  - add a lightweight safeguard so `no_progress_stalled` objective acceptance only triggers when edit history in the stage is non-trivial (avoid accepting pathological read-only stalls).
  - add guardrails for whitespace-collapsed write/edit payloads from local Qwen tool calls
  - add stricter timeout/no-progress handling for multi-step wandering tool loops
  - keep rerunning the same 8-task ladder until strict target-edit success reaches acceptance threshold before retrieval A/B phase
  - quantify precheck-hit rate and artifact-only-change rate in `summary.tsv` after the new harness changes
  - add targeted late-pipeline guidance for docs/contract tasks (task-8 class) so retries stop converging to check-failing partial docs edits

1. Tune loop budgets for overnight stability
- Keep new loop-profile controls stable (`normal`/`light`/`aggressive`/`auto`).
- Calibrate per-repo defaults for:
  - tool-loop rounds
  - repeat-call stall threshold
  - per-class repair budgets
- Extend current stop-reason-based `auto` recommendation into richer per-repo adaptive tuning (trend-aware and stage-aware, not only stop-reason counts).
- Keep newly added no-op turn-loop detector (`turn_no_progress_stalled`) tuned per repo profile and move threshold from static default to history-driven auto-tuning.

2. Improve summary quality for small-model memory
- Keep new structured schema stable (`objective`, `constraints`, `open_failures`, `focus`, `timeline`).
- Keep merged task-relevant carryover stable (multi-checkpoint trend baseline).
- Keep unresolved-failure lifecycle tracking stable (`resolved:` retirement path + unresolved carryover).
- Keep stage-aware trend weighting stable (`test`/`build`/`lint` impact ordering + recency/frequency blend).
- Add confidence scoring and longer-horizon decay policy so stale historical noise naturally de-prioritizes across many runs.

3. Verification pipeline hardening
- Keep newly added detection stable (`poetry`, `pnpm`, `uv`, `cargo`, `gradle`, `maven`, `bazel`, `tox`, `nox`).
- Calibrate stack-specific verify defaults where generic stages are weak (for example Maven/Bazel lint semantics).
- Stage-level timing and failure deltas now include severity scores; add adaptive thresholds and auto-escalation policy from run history.
- Tune per-stage no-progress streak thresholds from run history instead of one static cutoff.

## P1 — Claude Code Workflow Adaptation

1. LSP integration (next major capability)
- Keep stdio LSP bridge + fallback symbol context baseline.
- Keep the new cooldown/backoff health policy tuned for long AFK sessions.
- Keep persistent server/session reuse stable under long idle windows.
- Add richer capability mismatch diagnostics (definition/references support) in morning reports.
- Extend current lease-based cross-process supervision into a true daemon-owned reusable session host (IPC + heartbeat + restart recovery).

2. Coordinator hooks to real worker execution
- Keep new bounded `read`/`write`/`verify` worker-slot routing stable.
- Keep in-process queue telemetry stable (`submitted`/`completed`/`cancelled`/`rejected`/`pending`).
- Keep newly added write conflict-serialization and verify cancellation-request behavior stable overnight.
- Tune queue concurrency/pending presets per repo profile from prior run history.
- Add bounded worker queues + cancellation/escalation policy for AFK stalls.
- Keep long-lived isolated worker pool stable under overnight churn and repeated timeouts.
- Keep static hard-preemption knobs (`threshold` + `cooldown`) tuned per repo profile.
- Keep the new global run-history auto-tuning stable.
- Keep role-specific static thresholds stable (`read`/`write`/`verify`).
- Keep lane-specific (`read`/`write`/`verify`) adaptive threshold tuning stable under long AFK sessions.
- Keep lane-specific cooldown/backoff shaping stable and tune guardrails against over-cooling active lanes.
- Add worker-pool supervision policy tuning (health-check cadence + staggered respawn behavior) for long AFK runs.
- Keep preemption triage reporting stable (`why preempt happened`, `which lane triggered it most`).
- Expand causal tags with finer timeout sources (tool class and command family) for better overnight diagnosis.
- Add trend-based preemption summaries across multiple runs (day-over-day lane drift, worsening timeout classes).

3. Resume and report UX polish
- Keep checkpoint listing/inspection UX stable (`list-runs`, `show-run`).
- Keep `Unresolved blockers` severity ranking stable for long AFK sessions.
- Add concise `Required user input` detection in morning reports when retries are unlikely to converge.

4. CLI ergonomics and slash-command parity
- Keep Rust-first interactive chat path stable under long sessions and local small-model tool loops.
- Keep launcher-level hybrid routing stable (`chat` -> Rust, AFK `run/resume-run/list-runs/show-run` -> Python).
- Keep Python fallback chat parity stable with Rust for session semantics (ephemeral-by-default, explicit persistence opt-in only).
- Keep prompt-toolkit event-loop/input-state hardening stable in fallback mode (no read-only filter regressions).
- Tune fallback auto-compaction thresholds for long chat sessions so pre-turn compaction remains conservative but prevents context-overrun stalls.
- Continue closing command-surface parity gaps between Rust interactive slash commands and Python AFK control commands.
- Keep default ephemeral interactive sessions stable and retain explicit opt-in persistence (`CLAW_PERSIST_SESSION=1`).
- Keep newly added executable workflow commands (`/branch`, `/worktree`, `/commit`, `/pr`, `/issue`) stable under mixed repo states and permission modes.
- Keep `/commit-push-pr` stable after wiring (conservative preflight + existing commands-crate flow), and harden diagnostics for remote/auth failures.
- Keep newly wired Rust `/branch` and `/worktree` behavior stable across mixed repo states, especially read-only-mode gating and clear operator diagnostics.
- Keep `/budget` parity behavior stable across live REPL and `--resume` execution paths, and continue alias coverage tuning (`/resume`, `/checkpoint`) only where it reduces operator friction.
- Keep `/checkpoint` parity behavior stable (list/detail rendering, git-root checkpoint discovery, invalid-limit handling) across mixed repo states.
- Keep direct checkpoint command parity stable (`/list-runs`, `/show-run`) with consistent output and argument validation behavior.
- Keep REPL unknown-slash passthrough behavior stable so absolute path prompts (for example `/home/...`) remain usable without command-parser false positives.
- Keep launcher chat-surface routing stable (`--chat-surface`, `--python-ui`, `--rust-ui`, `CLAW_CHAT_SURFACE`) and avoid regressions in mode/arg parsing.
- Keep per-turn context/token telemetry stable in `/status` + `/cost` while extending parity with upstream statusline style.
- Continue replacing mirrored command stubs with executable Python implementations for high-value slash paths.
- Curate upstream `ultraworkers/claw-code` cherry-picks for CLI/REPL-only UX fixes while excluding Anthropic-first provider/auth assumptions.
- Upstream intake batch 1 completed (2026-04-22): `4cb8fa0` (code-only), `f3f6643`, `a3270db`, `47aa1a5`, `60ec2ae`, plus fork-compat patch to keep shorthand prompt behavior for non-flag multi-word input.
- Upstream intake batch 2 completed (2026-04-22): `4f670e5`, `cf129c8`, `3168642`, `78dca71`, `7587f2c`, `7ec6860`, `3ed27d5`, `11e2353`; resumed `/diff` JSON parity was merged with fork-local `session_path.parent()` context preserved.
- Next upstream intake target: operator UX polish batch (`7763ca3`, `79352a2`, `541c5bb`) with the same provider/auth scope guard.

## P2 — Runtime and Tooling Hardening

1. Semantic index runtime hardening
- Add explicit profile presets for embedding runtime/model choices.
- Add index metadata inspection command and rebuild recommendation command.
- Add safer mismatch handling for stale indexes across model/profile changes.

2. Tool safety envelopes
- Command denylist/allowlist profiles.
- File-write scope controls and risky-command escalation.

3. Observability
- Structured run logs for loops, retries, and regressions.
- Lightweight metrics for completion quality vs wall-clock time.

4. Small-model runtime presets
- Keep default runtime/model unchanged (`qwen3.5:4b`, localhost:8080).
- Track optional 2B experiment profile via `scripts/run_qwen35_2b_agentic_profile.sh`.
- Keep `--flash-attn` hardware-conditional:
  - older GPUs (for example RTX 2060): prefer `off`
  - newer GPUs: test `on` when stable
- Track upstream Qwen3.5 template fixes and retire local cache-reuse patch once GGUF/model metadata ships corrected template by default.

## Success Criteria

1. User can launch an overnight task and get a usable morning report.
2. Agent recovers from common failures without manual intervention.
3. Small-model limitations are offset by iterative correction loops.
