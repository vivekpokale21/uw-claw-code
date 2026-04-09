# Qwen Capability Matrix Post-Patch (2026-04-09)

Runtime:
- model: `qwen3.5:4b`
- endpoint: `LLM_BASE_URL=http://127.0.0.1:8129`
- CLI: `rust/target/debug/claw` (post Qwen compatibility patch)
- tools: `bash,read,write,edit,glob,grep`

Execution:
- dataset repo: external test dataset (`/home/vivek/projects/dubai_boom_monitor`)
- method: isolated git worktree per task (`/tmp/dbm_qwen_matrix_post_tN`) with a fresh `claw` process per task
- artifact root: `/tmp/qwen_matrix_postpatch_2026-04-09`

## Task Summary

| Task | Prompt shape | Exit | Stream/content fatal | Changed files (non-session) | Strict success |
|---|---|---|---|---|---|
| 1 | `.gitignore` entries | 0 | no | `.gitignore` | yes |
| 2 | single-file helper edit | 124 | no | `app/processor.py` | no (timed out + malformed code changes) |
| 3 | config + API two-file wiring | 0 | no | `app/config.py`, `app/fix_config.py` | no (partial + unexpected file) |
| 4 | API + DB optional filter | 0 | no | `app/db.py` | no (partial, API wiring missing) |
| 5 | cache-key scope update | 124 | no | `app/cache.py` | no (timed out + destructive formatting collapse) |
| 6 | GeoJSON derived property | 0 | no | `app/geojson.py` | yes |
| 7 | multi-file refactor extraction | 124 | no | `app/time_filters.py` | no (timed out + partial extraction) |
| 8 | endpoint + README update | 0 | no | `app/api.py`, `README.md` | yes |

## Result Delta vs Baseline

- Baseline strict success: `1/8`
- Post-patch strict success: `3/8`
- Stream/content fatal failures: unchanged at `0/8`

## Residual Failure Modes

- Long wandering tool loops on multi-step tasks (timeout-prone).
- Partial completion where only one of several required file changes is applied.
- Whitespace-collapsed payloads in write/edit arguments causing destructive formatting.

## Decision

- Qwen compatibility patch improved executable tool-call behavior materially, but capability gate still fails.
- Retrieval A/B remains deferred until strict task success improves further from `3/8`.
