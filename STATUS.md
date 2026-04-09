# STATUS.md

Last updated: 2026-04-09

## Migration Slices (AFK Local-First Port)

### Slice 1 — Rust local defaults + provider/base-url behavior (2026-04-09)

- Rust CLI default model switched to `qwen3.5:4b`.
- OpenAI-compatible default base URL switched to `http://127.0.0.1:8080/v1`.
- `LLM_BASE_URL` now has precedence over `OPENAI_BASE_URL` on OpenAI-compatible path.
- OpenAI-compatible local endpoints now allow missing API key when base URL is localhost.
- Provider detection now prefers OpenAI-compatible routing for local-model families (`qwen*`, `llama*`, `mistral*`, `deepseek*`, `phi*`) and `model:tag` style IDs (for example `qwen3.5:4b`), while keeping explicit prefix routing behavior.

### Slice 2 — Rust AFK controls/parity commands + run inspection (2026-04-09)

- Rust slash-command parser now supports `/budget`, `/checkpoint`, `/list-runs`, and `/show-run`.
- Rust REPL dispatch now implements:
  - `/budget` context/token budget report
  - `/checkpoint [list|<run-id>|list <limit>]`
  - `/list-runs [limit]`
  - `/show-run <run-id>`
- Run-inspection reads checkpoint artifacts from `<git-root>/.port_sessions/autonomous_<run-id>.json`.
- Unknown slash commands in the REPL now passthrough to prompt text (for example absolute path prompts like `/home/...`) instead of being hard-rejected as command errors.

### Slice 3 — Rust loop reliability bounds + deterministic stall reasons (2026-04-09)

- Runtime conversation loop now enforces deterministic stop reasons for reliability guardrails:
  - `stop_reason=tool_loop_stalled` when identical tool call+input repeats past configured threshold.
  - `stop_reason=no_progress_stalled` when identical tool plan repeats across iterations past configured threshold.
- Added configurable runtime guards (env-driven defaults):
  - `CLAW_TOOL_LOOP_STALL_LIMIT` (default `4`)
  - `CLAW_NO_PROGRESS_STALL_LIMIT` (default `3`)
- Rust REPL loop now prints pre-turn context budget status and runs pre-turn auto-compaction when context exceeds threshold.
- Added pre-turn auto-compaction reliability path using model-aware context thresholds (`AUTO_COMPACT_TRIGGER_PERCENT=85`) before model calls.

### Slice 4 — Rust LSP retrieval health persistence (2026-04-09)

- Rust LSP runtime now tracks per-language health telemetry with deterministic cooldown/backoff fields:
  - `consecutive_failures`
  - `blocked_until_unix`
  - `last_error`
  - `last_attempt_unix`
  - `last_success_unix`
  - `total_attempts`
  - `total_failures`
  - `last_warning`
  - `last_capabilities`
  - `last_failure_kind`
  - `recent_crash_loops`
- Rust LSP dispatch now enforces cooldown stop behavior before repeated failing requests and reports explicit cooldown errors.
- Rust LSP dispatch now supports `health`/`status` actions for structured health + server snapshots.
- Rust tools `LSP` execution now auto-loads and persists health state at:
  - default: `<repo-root>/.port_sessions/lsp_health_state.json`
  - override: `CLAW_LSP_HEALTH_STATE_FILE` or `LSP_HEALTH_STATE_FILE`
- Rust tools `LSP` schema now includes `health` and `status` actions.

### Slice 5 — Rust CLI UX polish (statusline, streaming, slash parity) (2026-04-09)

- Rust interactive pre-turn statusline now includes richer operator context:
  - context utilization percentage
  - pressure band (`low`/`medium`/`high`)
  - message/turn counters
  - compact session id label
- Rust slash parsing parity now includes:
  - `/runs [limit]` alias for `/list-runs [limit]`
  - numeric shorthand `/checkpoint <limit>` mapped to `/checkpoint list <limit>`
- Added Rust regression coverage for:
  - new statusline formatter pressure thresholds
  - slash alias + checkpoint shorthand parser behavior

### Slice 6 — Test hardening + Python compatibility cleanup notes (2026-04-09)

- Added Rust tool-level regression coverage for LSP health persistence contract:
  - verifies `LSP` tool emits `lsp_health_state_path`
  - verifies `.port_sessions/lsp_health_state.json` is written
  - verifies repeated failing dispatch updates persisted `health.rust.total_failures`
- Migration policy is now explicitly documented as Rust-primary / Python-fallback:
  - primary AFK workflow behavior remains on Rust path
  - Python path is compatibility-only where parity is incomplete
  - no new Python-first AFK feature development unless Rust path is blocked

## Current Status

Slice 6 of Claude Code workflow reconstruction for AFK reliability is now implemented on top of
`llama.cpp + qwen3.5:4b`.

Background framing for continuity:
- `claw-code` is being treated as a Claude Code fork/reimplementation effort after the source leak.
- Current focus remains high-value core workflow behavior for local small-model AFK operation.

## Completed in This Pass

1. Explicit autonomous workflow phases landed in `src/autonomous_run.py`:
- retrieve
- tool loop
- apply
- verify
- repair
- summarize

2. Deterministic stop reasons and AFK control behavior added:
- `complete`
- `verify_failed_budget_exhausted`
- `tool_loop_stalled`
- `max_iter_reached`
- `runtime_error`

3. Tool-first reliability mechanics added:
- retrieval evidence gate before completion
- bounded tool-loop rounds and repeated-call stall detection
- hierarchical memory compaction (`working` + `persistent`) used in model submissions

4. Verification/repair cycle upgraded:
- verify gate now runs detected `test` + `lint` + `build` commands
- failure classification + per-class repair budgets
- targeted repair prompts from verification output

5. Checkpoint/resume for autonomous runs added:
- persistent checkpoints in `.port_sessions/autonomous_<run_id>.json`
- new CLI command: `resume-run`
- morning report now records run id, checkpoint path, stop reason, phase history, verification stages, and repair attempts

6. Deterministic apply hardening:
- `apply_diff` now dry-runs patch before applying and returns structured error text on failure

7. Tests added/updated:
- new `tests/test_autonomous_workflow.py`
- all tests currently pass:
  - `python3 -m unittest discover -s tests -v`

8. Runtime-agnostic semantic embedding backend added:
- new shared embedding runtime client: `src/embedding_runtime.py`
- `agent/index/build.py` now resolves embedding runtime using:
  - `EMBEDDING_API_STYLE` (`openai` or `ollama`)
  - `LLM_BASE_URL` / `OPENAI_BASE_URL` defaults for openai-compatible mode
  - optional `OLLAMA_BASE_URL` compatibility mode
- `agent/index/search.py` now uses the same runtime resolver and validates query embedding dimension vs index metadata
- index builder now stores `index_meta` including embedding dim/model/runtime and supports dynamic embedding dimensions

9. LSP-style symbol retrieval integrated into active Python runtime:
- current repo check confirmed:
  - Rust workspace has a real LSP implementation under `rust/crates/lsp`
  - Python runtime had no active LSP path before this pass
- added Python-side symbol context resolver: `src/lsp_context.py`
- added local tool `symbol_context` in `src/tools.py` (definitions + references lookup)
- autonomous retrieve phase now runs symbol-context prefetch from task-derived candidates when retrieval evidence is missing
- retrieval gate now accepts `symbol_context` as a valid retrieval signal
- added true stdio JSON-RPC LSP bridge path:
  - configurable via `LSP_SERVER_COMMANDS_JSON` / `LSP_SERVER_COMMAND` / per-language env vars
  - uses `initialize`, `textDocument/didOpen`, `textDocument/definition`, and `textDocument/references`
  - falls back to local symbol index when LSP is unavailable or returns no signal
- added LSP usability hardening:
  - auto-discovery for common servers (`pyright-langserver`, `pylsp`, `typescript-language-server`, `rust-analyzer`, `gopls`)
  - new diagnostics command: `python3 -m src.main lsp-status`

10. LSP AFK reliability hardening added:
- LSP query failures are now tracked with health state (`consecutive_failures`, `last_error`, cooldown window).
- repeated failures trigger cooldown/backoff to avoid constant failing language-server launches overnight.
- symbol-context output now includes machine-readable LSP diagnostics (`lsp_attempted`, `lsp_health_key`).
- autonomous checkpoints now persist `lsp_health` snapshots.
- morning report now includes an `LSP health` section to explain retrieval instability quickly.
- tests added for:
  - cooldown skip behavior after launch failures
  - health section rendering in `lsp-status`
  - `lsp_health` persistence into checkpoint + morning report

11. LSP session reuse + capability diagnostics added:
- symbol retrieval now reuses persistent per-repo stdio LSP sessions (instead of process-per-query).
- session manager tracks idle TTL and evicts stale/dead sessions.
- document sync now uses `didOpen` + `didChange` across reused sessions.
- capability checks now gate requests:
  - skip `definition` when `definitionProvider` is unavailable
  - skip `references` when `referencesProvider` is unavailable
- capability mismatch diagnostics are emitted in symbol-context notes and reflected in LSP health snapshot fields.
- `lsp-status` now includes:
  - active session list (`alive`, `idle`, `requests`, capability summary)
  - enriched health details (`last_warning`, `last_caps`)
- tests added for:
  - session reuse across multiple symbol queries (single `Popen`)
  - capability mismatch path that avoids unsupported `references` calls

12. LSP session telemetry persisted in AFK artifacts:
- autonomous checkpoints now persist `lsp_sessions` snapshot alongside `lsp_health`.
- morning report now includes an `LSP sessions` section (`alive`, `idle`, `requests`, capability summary).
- autonomous workflow tests now assert `lsp_sessions` persistence and report rendering.

13. LSP crash-loop mitigation added:
- cooldown now applies deterministic jitter to reduce synchronized retry spikes.
- failures are classified (`startup`, `request`, `crash_loop`) in health telemetry.
- crash-loop failures from young/low-request sessions trigger an escalated cooldown multiplier.
- health snapshot and `lsp-status` now expose:
  - `last_failure_kind`
  - `recent_crash_loops`
- tests added for:
  - startup failure classification
  - crash-loop classification with escalated cooldown

14. Verification command detection expanded for mixed stacks:
- `_build_verification_commands` now supports:
  - `pnpm` script repos (when `pnpm-lock.yaml` is present)
  - `cargo` repos (`Cargo.toml`)
  - `gradle` repos (`build.gradle*` / `gradlew`)
  - `poetry` Python repos (`[tool.poetry]` in `pyproject.toml`)
  - `uv` Python repos (`uv.lock`)
- command priority remains deterministic:
  1) `make` target
  2) package scripts
  3) language/build-system defaults
- tests added for each new detection path to keep verify/repair loops stable across repo types.

15. Checkpoint inspection UX added:
- new CLI command: `list-runs --repo ... --limit N`
  - lists recent autonomous checkpoint files with status/stop reason/phase/iteration summary.
- new CLI command: `show-run <run_id> --repo ...`
  - prints the raw checkpoint JSON payload for detailed triage.
- added helper functions in `src/autonomous_run.py`:
  - `list_autonomous_checkpoints`
  - `render_autonomous_checkpoint_index`
  - `render_autonomous_checkpoint_detail`
- tests added for parser wiring and checkpoint helper rendering.

16. Configurable AFK loop budgets added:
- new run/resume CLI knobs:
  - `--repair-budget-per-class`
  - `--max-tool-rounds-per-turn`
  - `--repeat-call-limit`
  - `--verification-timeout-seconds`
- run loop now uses resolved `LoopBudgetConfig` instead of hardcoded thresholds.
- `detect_and_run_verification(...)` now accepts configurable timeout seconds.
- checkpoint payloads now persist `loop_budget` for reproducible resume behavior.
- morning report now includes `Loop budget` summary.
- tests added for:
  - parser support for all budget knobs
  - checkpoint/report loop-budget persistence
  - verification timeout propagation into verify gate

17. Verification stage timing telemetry added:
- each verification step now records `duration_seconds`.
- checkpoint `verification_steps` now persists stage timing.
- morning report verification section now includes per-stage timing (`label, Ns`).
- tests added to verify duration persistence and report rendering.

18. Verification failure-delta tracking added:
- run loop now tracks failure progress per stage across repair attempts:
  - `new_failure`
  - `no_change`
  - `changed`
- checkpoint now persists:
  - `verification_deltas`
  - `previous_failure_signatures`
- morning report now includes a `Verification deltas` section for quick AFK triage.
- tests added for delta persistence and report output.

19. Verification command detection expanded for additional ecosystems:
- `_build_verification_commands` now also supports:
  - `maven` repos (`pom.xml` / `mvnw`)
  - `bazel` repos (`WORKSPACE*` / `MODULE.bazel`)
  - `tox` repos (`tox.ini`)
  - `nox` repos (`noxfile.py`)
- tests added for each new stack detector to keep stage selection deterministic.

20. Verification regression severity scoring added:
- each verification delta now persists:
  - `failure_class`
  - `severity_score` (0-100)
  - `severity_level` (`low`/`medium`/`high`/`critical`)
  - `repair_attempt`
- repair prompts now include severity/delta context for tighter repair guidance.
- morning report now includes:
  - enriched `Verification deltas` lines (class + severity + score)
  - `Unresolved blockers` section ranked by severity.
- tests added for severity persistence and report rendering.

21. Coordinator worker-slot orchestration baseline added:
- autonomous run loop now routes work through bounded role slots:
  - `read`
  - `write`
  - `verify`
- each model submission now carries a coordinator worker hint in the system prompt:
  - `[coordinator worker]`
  - `worker_role=...`
  - `phase=...`
- new worker budget knobs (run/resume):
  - `--max-read-workers-per-run`
  - `--max-write-workers-per-run`
  - `--max-verify-workers-per-run`
- worker dispatch telemetry now persists in checkpoints:
  - `worker_budget`
  - `worker_counts`
  - `worker_dispatches`
- morning report now includes:
  - `Worker orchestration` summary
  - `Worker dispatches` history
- deterministic stop reason added for bounded slot exhaustion:
  - `worker_budget_exhausted`
- tests added for:
  - parser support for worker-slot knobs
  - worker telemetry persistence/report rendering
  - worker-budget exhaustion path
  - coordinator role-hint injection into submit system prompt

22. In-process coordinator worker queue added:
- new runtime module: `src/coordinator/runtime.py`
  - bounded per-role queues (`read`/`write`/`verify`)
  - per-role concurrency controls
  - pending-task cancellation API for AFK recovery
  - queue telemetry snapshots (`submitted`/`completed`/`cancelled`/`rejected`/`pending`)
- autonomous loop integration:
  - multi-read tool call batches now fan out through the read-worker queue in parallel
  - when `symbol_context` returns `has_signal=true`, pending read tasks are cancelled
  - queue-rejected read tasks degrade to deterministic synchronous fallback
- checkpoints now persist `worker_queue` telemetry.
- morning report now includes a `Worker queue` summary line.
- tests added:
  - `tests/test_coordinator_runtime.py` for parallelism, queue-limit rejection, cancellation behavior
  - autonomous integration tests for worker-queue persistence and parallel read-dispatch path

23. Write/verify worker lanes upgraded with queue-backed execution:
- tool loop now routes single-call execution through queue workers for all roles (`read`, `write`, `verify`) instead of direct inline-only execution.
- write-lane batching added:
  - non-conflicting write batches use parallel queue execution
  - conflicting write targets are forced into deterministic serialization with dispatch status `conflict_serialized`
- verify-lane batching added:
  - parallel verify tool calls now set cancellation intent after first failure
  - pending verify tasks are cancelled when possible and dispatch status records `verify_cancel_requested=<n>`
- queue fallback behavior remains deterministic:
  - queue rejection or worker exceptions degrade to synchronous tool execution
- tests added:
  - conflicting write batch serialization path
  - verify cancellation-request path after first failing verify tool result

24. Worker queue tuning controls added (run/resume):
- new CLI knobs:
  - `--read-worker-concurrency`
  - `--write-worker-concurrency`
  - `--verify-worker-concurrency`
  - `--max-pending-worker-tasks-per-role`
- new runtime resolver:
  - `WorkerQueueRuntimeConfig`
- autonomous checkpoints now persist `worker_queue_config`.
- morning report now includes worker queue config alongside queue stats.
- tests added for:
  - parser support for all queue tuning knobs
  - checkpoint persistence of `worker_queue_config`

25. Stronger preemption + optional process isolation added:
- new queue runtime controls:
  - `worker_task_timeout_seconds`
  - `worker_process_isolation`
- new CLI knobs (run/resume):
  - `--worker-task-timeout-seconds`
  - `--worker-process-isolation` / `--no-worker-process-isolation`
- autonomous queue collectors now enforce timeout windows and emit cancellation statuses:
  - `timeout_cancel_requested=<n>` (read lane)
  - `verify_timeout_cancel_requested=<n>` (verify lane)
  - `write_timeout_cancel_requested=<n>` (write lane)
- optional process isolation path:
  - new module `src/isolated_tool_worker.py`
  - new long-lived daemon + pool:
    - `src/isolated_tool_worker_daemon.py`
    - `src/coordinator/process_pool.py`
  - executes tool calls through a reusable isolated worker process pool and returns structured JSON output
  - used by autonomous loop when `worker_process_isolation=true`
- checkpoints now persist these settings under `worker_queue_config`.
- autonomous checkpoints and morning report now persist isolated pool health snapshot:
  - `enabled`
  - `pids`
  - `stats` (`requests`, `timeouts`, `dispatch_errors`, `respawns`)
- morning report worker-queue config text now includes timeout + process-isolation state.
- tests added:
  - parser coverage for new flags
  - checkpoint persistence for timeout/process-isolation fields
  - new `tests/test_isolated_tool_worker.py` for isolated worker payload handling and execution

26. Isolated worker pool supervision + explicit hard-preemption added:
- `src/coordinator/process_pool.py` now exposes:
  - `supervise_workers()` for proactive dead-worker replacement
  - `hard_preempt_all()` for deterministic pool-wide process reset on runaway timeout paths
- isolated pool telemetry now tracks:
  - `health_checks`
  - `dead_workers_replaced`
  - `hard_preemptions`
  - `workers_preempted`
- autonomous loop hardening in `src/autonomous_run.py`:
  - queue timeout paths now trigger explicit isolated pool hard-preemption events
  - isolated timeout tool messages now escalate to deterministic hard-preemption dispatch events
  - checkpoint/report snapshots now run worker supervision before persisting isolated pool status
- tests added:
  - process-pool supervision and hard-preemption coverage in `tests/test_isolated_tool_worker.py`
  - real process-isolation timeout escalation coverage in `tests/test_autonomous_workflow.py`

27. Isolated preemption policy tuning added (lane-safe escalation):
- new queue runtime controls:
  - `isolated_hard_preempt_timeout_threshold`
  - `isolated_hard_preempt_cooldown_seconds`
- new CLI knobs (run/resume):
  - `--isolated-hard-preempt-timeout-threshold`
  - `--isolated-hard-preempt-cooldown-seconds`
- autonomous loop now applies thresholded preemption policy:
  - timeout signals must meet threshold before hard-preempting the pool
  - cooldown prevents rapid repeated hard-preemption storms
  - timeout counters reset on successful non-timeout tool outcomes per role
- checkpoint persistence now includes `isolated_preempt_state`:
  - `consecutive_timeouts_by_role`
  - `last_preempt_epoch_seconds`
- tests added:
  - parser support for new preemption-policy knobs
  - checkpoint persistence for preemption-policy fields in `worker_queue_config`
  - real process-isolation integration test validating threshold-delayed hard preemption

28. Adaptive preemption defaults from run history added:
- when `worker_process_isolation=true` and preemption knobs are not explicitly passed, run startup now inspects recent autonomous checkpoints.
- new policy recommender derives default threshold/cooldown from recent isolated-pool telemetry (`timeouts`, `hard_preemptions`).
- tuned values are still persisted into checkpoint `worker_queue_config` for deterministic resume behavior.
- explicit CLI flags remain hard overrides (no auto-tune override when user values are provided).
- tests added:
  - policy recommendation unit coverage from synthetic checkpoint history
  - end-to-end run coverage proving auto-tuned values land in new checkpoint config

29. Morning-report preemption observability added:
- morning report now includes an `Isolated preempt state` line showing:
  - consecutive timeout counters by role
  - last hard-preemption epoch timestamp
- checkpoint persistence already included `isolated_preempt_state`; report output now surfaces it directly for AFK triage.
- tests added:
  - checkpoint + report coverage asserting `isolated_preempt_state` presence and report rendering.

30. Role-specific preemption thresholds added:
- preemption policy now supports per-lane timeout thresholds:
  - `read`
  - `write`
  - `verify`
- new CLI knobs (run/resume):
  - `--isolated-hard-preempt-timeout-threshold-read`
  - `--isolated-hard-preempt-timeout-threshold-write`
  - `--isolated-hard-preempt-timeout-threshold-verify`
- role-specific values override the global threshold while keeping shared cooldown behavior.
- worker-queue config checkpoint persistence now includes all role-specific threshold fields.
- morning report worker-queue config text now includes role-specific threshold values for AFK triage.
- tests added:
  - parser coverage for role-specific threshold flags
  - checkpoint persistence coverage for new fields
  - real process-isolation integration test proving role-specific verify threshold override behavior

31. Lane-specific adaptive threshold tuning from history added:
- run-history recommender now inspects worker dispatch telemetry per role (`read`/`write`/`verify`) in addition to global isolated-pool stats.
- auto-tuned defaults can now elevate only noisy lanes while leaving stable lanes at base threshold.
- explicit user-provided role/global threshold flags remain hard overrides.
- tests added:
  - policy recommendation coverage for role-specific tuned outputs
  - end-to-end run coverage validating auto-tuned role-specific threshold application in checkpoint config

32. Morning-report preemption triage summary added:
- morning report now includes `Preempt triage` with:
  - `top_role` (lane with strongest timeout/preempt signal)
  - per-lane timeout signal counts
  - per-lane hard-preemption counts
- this surfaces runaway-lane diagnosis directly in overnight report output without opening raw checkpoint JSON.
- tests added:
  - process-isolation timeout integration test now validates report triage rendering (`top_role=verify`).

33. Structured preemption decision events added:
- runtime now records structured `isolated_preempt_events` entries for each preemption decision:
  - `threshold_not_reached`
  - `cooldown_active`
  - `hard_preempted`
  - `hard_preempt_error`
- each event captures iteration, lane, reason, timeout count/threshold, cooldown remaining, and preempted worker count.
- checkpoints now persist `isolated_preempt_events` for deterministic AFK postmortems.
- morning report now includes a `Preempt events` section with recent decision entries.
- tests added:
  - timeout escalation integration coverage asserting event persistence and report rendering
  - threshold-delayed integration coverage asserting both decision phases are captured

34. Preemption event compaction summary added:
- checkpoints now persist `isolated_preempt_summary` with:
  - total event count
  - counts by decision
  - counts by lane
- morning report now includes a compact `Preempt summary` line before detailed event rows.
- this keeps long-run AFK diagnostics scannable even when event logs grow.
- tests added:
  - timeout escalation coverage asserts summary presence in checkpoint and report output
  - threshold-delayed coverage validates summary decision counts

35. Persistent memory compaction now enforces token-aware budget:
- `_compact_memory_layers(...)` now supports an explicit token budget (`max_tokens`) alongside char budget (`max_chars`).
- new default cap added:
  - `MAX_PERSISTENT_MEMORY_TOKENS = 1000`
- persistent summary truncation is now budget-driven:
  - enforces both char and approximate-token limits
  - still preserves newest context tail for AFK continuity
- this reduces long-run prompt bloat for `qwen3.5:4b` while keeping compacted memory deterministic.
- tests added:
  - `test_memory_compaction_respects_token_budget` validates budget enforcement and recency retention behavior.

36. Persistent memory compaction now preserves high-signal failure focus:
- `_compact_memory_layers(...)` now compacts into structured sections:
  - `focus: ...` (high-signal failures/blockers/repair markers)
  - `timeline: ...` (normal tool/action history)
- compaction logic now:
  - parses legacy flat summaries and upgrades them in-place
  - deduplicates entries while preserving latest signal
  - trims timeline first under budget pressure, then focus, to keep failure context visible
- this improves AFK reliability for small models by keeping unresolved failure context in prompt memory during long runs.
- tests added:
  - `test_memory_compaction_keeps_failure_focus_under_budget` verifies failure context survives aggressive token compaction.

37. Cross-run persistent-memory carryover added:
- new helper `_load_recent_persistent_summary(...)` seeds new autonomous runs from the most recent checkpoint that has non-empty `persistent_summary`.
- startup seeding is applied only for fresh runs (`resume-run` unchanged) and then normalized through existing memory compaction budget rules.
- checkpoints now persist `memory_seed` metadata:
  - `seeded`
  - `source_run_id`
- morning report now surfaces `Memory seed` for AFK handoff clarity.
- tests added:
  - `test_run_seeds_persistent_memory_from_latest_checkpoint` validates checkpoint seeding and report visibility.

38. Task-relevant carryover selection added:
- memory seeding now scores candidate checkpoints by lexical overlap with current task text.
- when overlap exists, seeding prefers the most task-relevant checkpoint over purely newest-by-timestamp.
- fallback remains deterministic:
  - newest non-empty checkpoint summary when no overlap signal is found.
- tests added:
  - `test_run_seeds_persistent_memory_from_task_relevant_checkpoint` validates relevance-based selection.

39. Retrieval-gate stall guard added to turn loop:
- autonomous loop now tracks consecutive retrieval-gate completion blocks.
- new deterministic stop path:
  - `retrieval_gate_stalled` after repeated completion attempts without retrieval evidence.
- checkpoint now persists `retrieval_gate_block_count`.
- morning report now includes `Retrieval gate blocks` for fast AFK triage.
- tests added:
  - `test_run_stops_when_retrieval_gate_repeatedly_blocks_completion` validates early stall stop, checkpoint field, and report line.

40. Verification no-progress stall guard added:
- verification loop now tracks per-stage consecutive `no_change` failure streaks from failure-signature deltas.
- new deterministic stop path:
  - `verify_no_progress_stalled` when the same stage fails repeatedly with unchanged output.
- checkpoint now persists `verification_no_change_streaks`.
- morning report now includes `Verification no-change streaks`.
- verification delta telemetry now includes `no_change_streak` per event.
- tests added:
  - `test_run_stops_when_verification_failure_has_no_progress_streak` validates stop reason + checkpoint/report telemetry.

41. LSP health now persists across process restarts:
- `src/lsp_context.py` now persists health state to repo-local runtime state:
  - `.port_sessions/lsp_health_state.json`
  - override supported via `LSP_HEALTH_STATE_FILE`
- persisted fields include cooldown windows, failure counters, failure kind, and capability diagnostics.
- runtime now reloads persisted health lazily per repo and keeps existing in-memory session reuse behavior unchanged.
- `lsp-status` now loads persisted health for the current working repo path.
- this is a reuse-first hardening step toward daemon-level reliability without rebuilding LSP stack internals.
- tests added:
  - `test_lsp_cooldown_persists_across_runtime_reset` validates cooldown survives in-memory reset and prevents immediate re-launch storm.

42. CLI chat mode with slash-command control landed:
- new interactive command added: `python3 -m src.main chat`.
- new chat-shell module: `src/cli_shell.py`.
- slash commands now available in chat mode:
  - `/help`
  - `/status`
  - `/thinking [on|off]`
  - `/clear`
  - `/tools [query]`
  - `/commands [query]`
  - `/lsp-status`
  - `/list-runs [limit]`
  - `/show-run <run_id>`
  - `/run <task>`
  - `/quit`
- plain chat input now uses a bounded tool-call loop that reuses existing local tools and deterministic tool execution path.
- run/resume parser now supports explicit reasoning toggle:
  - `--thinking` / `--no-thinking`
- model runtime payload now forwards chat-template thinking control:
  - `chat_template_kwargs.enable_thinking`
  - default is off unless explicitly enabled (`CLAW_ENABLE_THINKING` or CLI flag).
- context headroom is now surfaced in user-facing outputs:
  - run/resume stdout now prints `context_chars` and `context_chars_remaining`
  - checkpoints now persist `context_budget`
  - morning report now includes `Context budget`
- checkpoints now persist `thinking_enabled`; resume-run reuses persisted thinking mode unless explicitly overridden.
- tests added:
  - parser coverage for `chat` command and run/resume thinking flags
  - model-client payload/default thinking behavior
  - checkpoint/report context-budget persistence
  - chat-shell slash-command and tool-loop behavior (`tests/test_cli_shell.py`)

43. Reuse-first CLI parity slice delivered for interactive reliability:
- chat parser now routes only known slash commands; unknown slash-prefixed input is treated as normal prompt text.
  - this fixes absolute path prompts like `/home/vivek/projects/...` being rejected as unknown slash commands.
- chat now exposes a core ops command pack:
  - `/cost`, `/model`, `/permissions`, `/semantic`, `/session`, `/config`, `/memory`, `/diff`
  - existing commands retained (`/status`, `/thinking`, `/tools`, `/commands`, `/lsp-status`, `/run`, `/list-runs`, `/show-run`, `/clear`, `/quit`).
- semantic behavior in chat is now explicit and runtime-aware:
  - auto-attempt index bootstrap only when embeddings are available.
  - semantic tool schema is gated by chat semantic mode and index readiness.
  - `/semantic status|enable|disable|build|rebuild` added for manual evaluation flow.
- permissions mode is now enforced in chat tool execution:
  - `read-only` blocks `run_command` and `apply_diff`.
  - `workspace-write` / `danger-full-access` retain write-capable behavior (with existing `allow_apply_diff` gate).
- managed chat sessions added under `.port_sessions/chat`:
  - `/session list` and `/session switch <id>` persist and restore message history + model/thinking/usage state.
- model client now normalizes and exposes usage tokens from OpenAI-compatible responses:
  - `input_tokens` / `output_tokens` carried into chat `/status` and `/cost`.
- interactive UX now supports richer terminal mode with optional dependencies:
  - `prompt_toolkit` + `rich` integration
  - explicit plain fallback via `chat --plain`.
- tests added/updated:
  - `tests/test_cli_shell.py`
    - unknown-slash path fallback
    - semantic tool-schema gating via `/semantic disable`
    - read-only permission enforcement for tool loop
    - session list/switch state restore
  - `tests/test_model_defaults_and_benchmark.py`
    - `chat --plain` parser coverage
    - model-client usage-token normalization coverage

44. CLI parity expanded with high-value operational aliases + transcript controls:
- added new chat slash commands:
  - `/budget` (context/tool budget snapshot)
  - `/compact [keep_recent]` (deterministic conversation compaction)
  - `/export [path]` (Markdown transcript export)
  - `/version` (local runtime + git metadata snapshot)
- added AFK-focused aliases to reduce command friction:
  - `/resume [session_id]` -> session list/switch
  - `/checkpoint [list|<run_id>]` -> checkpoint list/detail
- all new commands reuse existing Python runtime surfaces for sessions/checkpoints and avoid introducing new provider/runtime coupling.
- tests added/updated:
  - `tests/test_cli_shell.py`
    - compact behavior + summary insertion
    - export path write + transcript content checks
    - budget/version command output checks
    - resume/checkpoint alias behavior
  - `tests/test_cli_shell_integration.py`
    - real git-repo integration coverage for `/diff` command output path

45. Reuse-first workflow command parity landed in chat shell:
- added executable slash commands backed by local git/gh tooling:
  - `/branch [list|create <name>|switch <name>]`
  - `/worktree [list|add <path> [branch]|remove <path>|prune]`
  - `/commit <message>`
  - `/pr <title>`
  - `/issue <title>`
- behavior is deterministic and permission-aware:
  - `read-only` now blocks write-capable workflow commands (`/branch create|switch`, `/worktree add|remove|prune`, `/commit`, `/pr`, `/issue`).
  - read-only-safe actions remain available (`/branch list`, `/worktree list`).
- implementation remains reuse-first:
  - commands are wrappers around existing system tools (`git`, `gh`) with structured, parseable output and explicit failure reasons.
- tests added/updated:
  - `tests/test_cli_shell.py`
    - branch list command wiring
    - worktree add branch-exists fallback path
    - commit usage validation
    - read-only blocking for workflow commands
    - `gh` missing-path behavior for `/pr` and `/issue`
  - `tests/test_cli_shell_integration.py`
    - real git integration for `/branch create` + `/commit`

46. Tool-loop reliability hardening shipped for chat + autonomous AFK loop:
- chat tool-loop fallback is now deterministic:
  - chat default `--max-tool-rounds` raised from `8` to `12`.
  - when the cap is reached, chat now performs one forced no-tools finalization submit and returns direct answer text when available.
  - fallback error text now includes repeated tool-call summary when finalization still fails.
- autonomous no-op turn guard landed:
  - new loop budget knob: `max_no_progress_turn_streak` (run/resume + checkpoint persistence).
  - new stop reason: `turn_no_progress_stalled`.
  - checkpoint now persists `turn_no_progress_streak`.
  - morning report now includes `No-progress turn streak`.
- tests added/updated:
  - `tests/test_cli_shell.py`
    - forced-finalization path after chat tool-loop cap.
  - `tests/test_autonomous_workflow.py`
    - parser support for `--max-no-progress-turn-streak` (run/resume)
    - checkpoint loop-budget persistence for new field
    - stop behavior coverage for `turn_no_progress_stalled`
  - `tests/test_model_defaults_and_benchmark.py`
    - parser default check for chat `max_tool_rounds=12`

47. Loop-profile presets + history-based auto budget recommendation added:
- run/resume now support `--loop-profile`:
  - `normal`
  - `light`
  - `aggressive`
  - `auto`
- loop profile now participates in loop-budget resolution with deterministic precedence:
  - explicit CLI budget flags
  - selected loop-profile overrides
  - checkpoint budget (resume)
  - defaults
- `auto` now derives baseline loop-budget recommendations from recent checkpoint stop reasons (for example tool-loop stalls, verify-budget exhaustion, no-progress stalls, runtime errors).
- checkpoints now persist:
  - `loop_profile`
  - `loop_budget_recommendation`
- morning report now includes `Loop profile` alongside `Loop budget`.
- tests added/updated:
  - parser coverage for `--loop-profile` (run/resume)
  - profile budget application coverage (`aggressive`)
  - recommendation helper coverage from synthetic checkpoint history

48. Persistent-memory schema + cross-run trend carryover upgraded:
- memory compaction now supports and preserves a structured schema:
  - `objective`
  - `constraints`
  - `open_failures`
  - `focus`
  - `timeline`
- autonomous runs now always initialize structured memory with baseline constraints:
  - model id
  - `apply_diff` policy
  - retrieval-first guard flag
- cross-run memory carryover now merges multiple task-relevant checkpoint summaries (top scored recent matches), instead of selecting only one checkpoint summary.
- carryover source run id now points to the latest merged relevant run for clearer morning-report provenance.
- tests added/updated:
  - structured schema compaction coverage
  - relevant multi-checkpoint trend merge coverage
  - run-level initialization coverage for `objective` + `constraints`

49. Persistent-memory lifecycle + trend weighting follow-up landed:
- open-failure lifecycle now supports explicit resolution markers:
  - entries prefixed with `resolved:` are treated as closed failures
  - resolved failures are retired from `open_failures` during compaction/merge
- cross-run carryover now applies stage-aware trend weighting for unresolved blockers:
  - scoring combines recency, stage weight (`test` > `build` > `lint` > other), and repeat frequency
  - merged carryover prioritizes likely-impactful recent failures over stale lint noise
- memory compaction preserves ranked carryover order while still applying lifecycle filtering and token budget constraints.
- tests added/updated:
  - resolved-failure retirement behavior in merged summaries
  - stage-aware prioritization behavior for recent test failures vs stale lint failures

50. LSP daemon lease supervision hardening landed:
- added daemon lease lifecycle hardening in `src/lsp_context.py`:
  - lease release now covers owned-repo cases even when no session object is active
  - cooldown-skip path no longer acquires daemon lease unnecessarily
  - startup/request failure paths release daemon lease when no reusable session survives
- added helper paths for lease refresh/release based on active-session reality:
  - `_has_active_lsp_session_for_repo(...)`
  - `_refresh_or_release_lsp_daemon_lease(...)`
- tests added:
  - `test_close_all_lsp_sessions_releases_owned_daemon_lease_without_sessions`
  - `test_lsp_cooldown_skip_does_not_acquire_daemon_lease`
  - `test_lsp_startup_failure_releases_daemon_lease_when_no_session_survives`

51. Coordinator lane-specific adaptive cooldown shaping expanded:
- `_recommend_isolated_preempt_policy_from_history(...)` now consumes historical `isolated_preempt_events`.
- per-lane cooldown recommendation now includes `cooldown_active` pressure in addition to timeout/hard-preempt counts.
- behavior effect:
  - lanes that repeatedly hit cooldown gates are assigned higher cooldowns on next runs to reduce preempt thrash.
- test added:
  - `test_run_auto_tunes_role_specific_cooldown_from_preempt_event_history`

52. Interactive CLI parity follow-up landed:
- added live interactive statusline rendering (`model`, `thinking`, `permissions`, `context`, `messages`, `semantic`):
  - prompt-toolkit mode via bottom toolbar
  - plain mode via pre-prompt status row
- added streamed incremental assistant output rendering path in interactive chat mode.
- unit tests added:
  - `test_live_statusline_contains_context_and_mode`
  - `test_prompt_can_stream_output_chunks_and_suppress_return`

53. Qwen3.5 cache-reuse template mitigation landed for llama.cpp startup:
- added patched chat template artifact:
  - `runtime_templates/qwen35_chat_template_cachefix.jinja`
  - one-line guard: historical `<think>...</think>` wrapper now requires non-empty `reasoning_content`
- runtime bootstrap now injects template directly into llama.cpp startup env:
  - `LLAMA_ARG_CHAT_TEMPLATE_FILE=<patched-template>`
  - default enabled, with controls:
    - `CLAW_QWEN35_TEMPLATE_FIX=0` (disable)
    - `CLAW_QWEN35_CHAT_TEMPLATE_FILE=/path/to/template.jinja` (override)
- one-command launcher (`scripts/start_claw_cli.sh`) now applies the same default injection path.
- tests added:
  - `test_augment_runtime_env_skips_template_injection_when_fix_disabled`
  - existing startup bootstrap test now asserts template-file env injection on auto-start path.

54. Split-pane interactive CLI parity slice landed (reuse-first over existing `ChatShell` flow):
- prompt-toolkit mode now runs as a full-screen TUI with explicit panes:
  - `Conversation` (left)
  - `Status` sidebar (right; session + context/token budget + usage + hints)
  - `Input` (bottom)
- assistant replies now stream incrementally into the conversation pane while turns execute in a background thread.
- UI remains responsive during active turns via async update pump + queued stream/result events.
- keyboard controls added:
  - `Ctrl-C` exit
  - `Ctrl-L` clear conversation pane
- tests added:
  - `test_tui_sidebar_contains_session_budget_usage_and_hints`

55. Rust-first interactive CLI migration slice landed (hybrid with Python AFK loop):
- launcher routing now targets Rust interactive chat by default while keeping AFK Python commands via same script:
  - `chat` (default) -> Rust CLI
  - `run`, `resume-run`, `list-runs`, `show-run` -> Python CLI passthrough
- Rust runtime defaults now align with local qwen/llama.cpp target:
  - default model changed to `qwen3.5:4b`
  - default permission mode changed to `workspace-write`
  - OpenAI-compatible provider path now supports:
    - `LLM_BASE_URL` precedence over `OPENAI_BASE_URL`
    - localhost-first default base URL
    - local no-key mode (no `OPENAI_API_KEY` required for localhost endpoints)
    - `/v1/chat/completions` normalization when base URL omits `/v1`
- interactive context reliability hardening in Rust:
  - per-turn context budget status line (`estimated/max/remaining`)
  - proactive auto-compaction when context budget nears saturation
  - one-retry structured-tool protocol guard when model emits XML-like `<tool_call>` text instead of structured tool calls
- session persistence behavior changed for interactive Rust path:
  - sessions are ephemeral by default (`CLAW_PERSIST_SESSION=0`)
  - cross-restart context carryover is opt-in (`CLAW_PERSIST_SESSION=1`)
- tests added/updated in Rust crates for:
  - OpenAI-compatible local runtime auth/base-url behavior
  - structured-tool protocol retry detection
  - status report context budget fields

56. Python fallback interactive reliability parity follow-up landed:
- session persistence policy now matches Rust interactive behavior:
  - default is ephemeral process-scoped session storage
  - cross-restart persistence is opt-in only (`CLAW_PERSIST_SESSION=1`)
- Python interactive status outputs now expose persistence mode (`ephemeral` vs `on`) in `/status` and the split-pane sidebar.
- prompt-toolkit read-only handling now uses a filter-safe setter to avoid runtime crashes:
  - fixed `'bool' object is not callable` failure during background turn completion/unlock.
- tests added/updated:
  - `test_chat_shell_defaults_to_ephemeral_session_dir_when_persistence_disabled`
  - `test_chat_shell_uses_repo_session_dir_when_persistence_enabled`
  - `test_set_text_area_read_only_uses_filter_wrapper`
- validation:
  - `python3 -m unittest discover -s tests -v` (`Ran 154 tests`, `OK`)

57. Python chat context compaction guardrail landed:
- added proactive pre-turn auto-compaction in `ChatShell` prompt path:
  - trigger threshold: ~85% of context budget
  - applies bounded compaction passes before model submit
  - increments `auto_compaction_count` telemetry for session-level observability
- goal: prevent tool-loop dead-ends from oversized chat payloads during long unattended sessions.
- tests added:
  - `test_auto_compact_before_prompt_when_context_near_budget`
  - `test_auto_compact_before_prompt_skips_when_context_is_low`
- validation:
  - `python3 -m unittest discover -s tests -v` (`Ran 156 tests`, `OK`)

58. Rust `/commit-push-pr` command is now wired in REPL with reuse-first implementation:
- removed placeholder path (`commit-push-pr not yet wired`) and connected REPL command dispatch to existing `commands` crate workflow handler:
  - `handle_commit_push_pr_slash_command(...)`
  - no duplicate branch/push/PR orchestration logic added in `claw-cli`.
- added conservative preflight guard before commit/push/PR:
  - block detached HEAD state
  - block unresolved merge conflicts (`git diff --name-only --diff-filter=U`)
- added read-only permission guard for `/commit-push-pr`.
- tests added in Rust `claw-cli` module:
  - `commit_push_pr_preflight_passes_on_clean_named_branch`
  - `commit_push_pr_preflight_rejects_detached_head`
- validation:
  - `cargo test -p claw-cli -- --test-threads=1` (`79 passed`, `0 failed`)

59. Rust `/branch` + `/worktree` REPL parity wiring landed (reuse-first):
- removed REPL placeholder paths for `/branch` and `/worktree`.
- wired both commands to existing shared handlers from `commands` crate:
  - `handle_branch_slash_command(...)`
  - `handle_worktree_slash_command(...)`
- added read-only-mode write gating:
  - `/branch create|switch` blocked in `read-only`
  - `/worktree add|remove|prune` blocked in `read-only`
  - list/read actions remain allowed.
- added Rust unit tests for write-action classification helpers:
  - `branch_write_permission_helper_flags_mutating_actions`
  - `worktree_write_permission_helper_flags_mutating_actions`
- validation:
  - `python3 -m unittest discover -s tests -v` (`Ran 156 tests`, `OK`)
  - `cargo test -p commands -- --test-threads=1` (`16 passed`, `0 failed`)
  - `cargo test -p claw-cli -- --test-threads=1` (`79 passed`, `0 failed`)

60. Rust `/budget` slash-command parity landed:
- shared command surface updated:
  - `commands` crate slash-command spec now includes `/budget` with resume support metadata.
  - parser + help rendering tests updated to cover `/budget`.
- runtime wiring updated:
  - interactive REPL now handles `/budget` with structured budget output.
  - `--resume SESSION /budget` now returns budget report parity with live REPL behavior.
- tests updated:
  - REPL help coverage now asserts `/budget` presence.
  - resume-supported command list test now includes `budget`.
- validation:
  - `cargo test -p commands -- --test-threads=1` (`16 passed`, `0 failed`)
  - `cargo test -p claw-cli -- --test-threads=1` (`79 passed`, `0 failed`)
  - `python3 -m unittest discover -s tests -v` (`Ran 156 tests`, `OK`)

61. Rust interactive slash-path UX parity landed:
- fixed REPL slash parsing behavior to match Python fallback ergonomics:
  - unknown slash-prefixed input now falls through to normal model prompt execution.
  - absolute paths (for example `/home/...`) are no longer blocked as unknown slash commands.
- implementation detail:
  - added `parse_repl_slash_command(...)` wrapper that suppresses `SlashCommand::Unknown(...)` in REPL-only parsing.
  - direct slash CLI mode (`claw /...`) and `--resume` command parsing remain unchanged.
- tests added:
  - `repl_slash_parser_ignores_unknown_commands`
- validation:
  - `cargo test -p commands -- --test-threads=1` (`16 passed`, `0 failed`)
  - `cargo test -p claw-cli -- --test-threads=1` (`80 passed`, `0 failed`)
  - `python3 -m unittest discover -s tests -v` (`Ran 156 tests`, `OK`)

62. Rust `/checkpoint` slash-command parity landed:
- shared slash-command manifest/parser now includes:
  - `/checkpoint [list|<run-id>|list <limit>]`
- interactive runtime wiring added:
  - `/checkpoint` or `/checkpoint list [limit]` lists `.port_sessions/autonomous_*.json` checkpoints from git-root scope.
  - `/checkpoint <run-id>` renders the selected checkpoint payload (pretty JSON) and surfaces a deterministic not-found message when missing.
- parity behavior notes:
  - index output mirrors existing Python AFK tooling layout (`Checkpoint files: N`, tabular header with run metadata).
  - invalid list limits return a usage-guided error instead of terminating the REPL.
- tests updated:
  - commands parser/help tests for `/checkpoint` surface
  - REPL help + slash parse coverage in `claw-cli` tests
- validation:
  - `cargo test -p commands -- --test-threads=1` (`16 passed`, `0 failed`)
  - `cargo test -p claw-cli -- --test-threads=1` (`80 passed`, `0 failed`)
  - `python3 -m unittest discover -s tests -v` (`Ran 156 tests`, `OK`)

63. Rust direct checkpoint command parity landed (`/list-runs`, `/show-run`):
- shared slash-command surface expanded:
  - `/list-runs [limit]`
  - `/show-run <run-id>`
- runtime wiring:
  - `/list-runs` now reuses checkpoint index rendering used by `/checkpoint list`.
  - `/show-run` now reuses checkpoint detail rendering used by `/checkpoint <run-id>`.
  - missing `run-id` now returns deterministic usage guidance (`Usage: /show-run <run-id>`).
- tests updated:
  - commands parser/help coverage for `/list-runs` and `/show-run`
  - REPL help + parser coverage in Rust `claw-cli` tests
- validation:
  - `cargo test -p commands -- --test-threads=1` (`16 passed`, `0 failed`)
  - `cargo test -p claw-cli -- --test-threads=1` (`80 passed`, `0 failed`)
  - `python3 -m unittest discover -s tests -v` (`Ran 156 tests`, `OK`)

64. Launcher chat-surface selector landed (reuse over rebuild):
- updated `scripts/start_claw_cli.sh` to support explicit chat UI/runtime surface routing:
  - `--chat-surface rust|python`
  - `--rust-ui` / `--python-ui` shortcuts
  - `CLAW_CHAT_SURFACE` env fallback
- behavior:
  - default remains Rust interactive chat path.
  - Python chat surface (split-pane prompt-toolkit UI) can now be selected directly from launcher without manual script edits.
  - Rust toolchain absence now auto-falls back to Python by switching `CHAT_SURFACE=python`.
- validation:
  - `bash -n scripts/start_claw_cli.sh` (`OK`)
  - `python3 -m unittest discover -s tests -v` (`Ran 156 tests`, `OK`)

## Known Gaps

1. LSP now has cross-process lease supervision, but reusable LSP sessions are still process-local/in-memory (no shared daemon IPC session host yet).
2. Coordinator process-isolation now has explicit hard preemption + supervision, but preemption granularity is still worker-process level (not per-subtask kill inside a running worker process).
3. Persistent memory now includes structured schema, lifecycle retirement, and stage-aware trend ranking, but it still lacks confidence scoring and longer-horizon decay (for example week-over-week failure drift weighting).
4. Turn-loop no-op detection now exists (`turn_no_progress_stalled`), but tuning is still static and not yet auto-calibrated per repo profile.
5. Semantic index still assumes a single embedding model per built index; switching models requires rebuild and there is no automatic migration flow.
6. Chat-mode slash commands currently map to implemented Python surfaces; mirrored archive command entries are still mostly metadata/stub execution paths.

## Next Step

Implement Slice 2 adapters:
1. LSP execution hardening (cross-process daemon supervision + restart persistence for language servers).
2. Coordinator runtime hardening (add richer lane-specific adaptive backoff curves and cooldown shaping, not just threshold bumps).
3. Small-model memory quality hardening follow-up (confidence scoring + longer-horizon decay policy on top of the new structured lifecycle/trend baseline).
4. Semantic index UX hardening (provider presets, clear rebuild prompts, and model/profile checks).
