# Qwen Capability Matrix Baseline (2026-04-09)

Runtime:
- model: `qwen3.5:4b`
- endpoint: `LLM_BASE_URL=http://127.0.0.1:8129`
- CLI: `rust/target/debug/claw` (`Git SHA 938dfee`)
- tools: `bash,read,write,edit,glob,grep`

Execution:
- dataset repo: external test dataset (`/home/vivek/projects/dubai_boom_monitor`)
- method: isolated git worktree per task (`/tmp/dbm_qwen_matrix_cwd_tN`), with `claw` launched from each target worktree `cwd`
- artifact root: `/tmp/qwen_matrix_baseline_2026-04-09_cwd`

## Task Summary

| Task | Prompt shape | Stream/content fatal | Target worktree edits | py_compile check |
|---|---|---|---|---|
| 1 | `.gitignore` exact entries | no | `.gitignore` | pass |
| 2 | single-file helper edit | no | none | pass |
| 3 | config + API two-file wiring | no | none | pass |
| 4 | API + DB optional filter | no | none | pass |
| 5 | cache-key scope update | no | none | pass |
| 6 | GeoJSON derived property | no | none | pass |
| 7 | multi-file refactor extraction | no | none | pass |
| 8 | endpoint + README update | no | none | pass |

## Observed Failure Pattern

- `assistant stream produced no content` did not recur in this run.
- Most runs returned no-op completions for the target worktree:
  - path/workspace confusion despite explicit repo path in prompt
  - pseudo tool-call text emission without completed edit loop
  - degraded text readability (space-collapsed prose blocks) in assistant output

## Decision

- Baseline capability gate failed (`1/8` successful target edits).
- Retrieval A/B phase is deferred until baseline edit reliability is improved.
