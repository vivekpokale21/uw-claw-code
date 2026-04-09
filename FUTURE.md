# FUTURE.md

Last updated: 2026-04-09

## Migration Slices (AFK Local-First Port)

### Completed

- Slice 1 (2026-04-09): Rust local runtime defaults and OpenAI-compatible base URL behavior now align with local llama.cpp-first operation (`qwen3.5:4b`, localhost `/v1`, `LLM_BASE_URL` precedence, local no-key allowance).

### Next

- Slice 2: Rust AFK command/control parity (`/checkpoint`, `/list-runs`, `/show-run`, `/budget`) plus unknown-slash passthrough in REPL.

## Priority Backlog

## P0 — AFK Reliability Core

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
