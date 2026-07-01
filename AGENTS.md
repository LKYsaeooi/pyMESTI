# AGENTS.md

This folder is an independent Python sub-project with its own Git repository and lightweight Codex handoff files.

## Start Here

1. Read `TASK_STATE.md` first.
2. Run `git status --short`.
3. Inspect only the files relevant to the current task.
4. Use Git diffs, focused tests, and existing code as the detailed memory.
5. Do not rely on previous chat context unless the user explicitly provides it.

## Work Rules

- Prefer small, focused edits.
- Do not rewrite unrelated files.
- Do not modify generated files, logs, datasets, checkpoints, virtual environments, or CodeGraph databases.
- Do not update the old v7 ledger during normal work.
- Treat `docs/legacy/mesti_python_port_plan_v7.md` as read-only historical context.
- Keep `TASK_STATE.md` under 100 lines.
- Do not create long progress summaries; reference files, functions, commands, and test results instead.

## Questions

Ask no more than 3 focused questions only when context is blocking.

A question is blocking only if answering it would change the implementation design, public API, data format, experiment protocol, or test expectation.

Do not ask questions about information that can be inferred from the repository.

## Before Ending

1. Run focused tests for source changes, or explain why tests were not run.
2. Run `git diff --stat`.
3. Run `git status --short`.
4. Update `TASK_STATE.md` compactly.
5. Append one short milestone entry to `TASK_LOG.md` only if a meaningful step was completed.

## Commits

- Make commits only when the user explicitly asks.
- If committing, use small commits with clear messages.
