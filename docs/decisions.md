# Decisions

## 2026-06-11 - Lightweight Git handoff

- Treat `Simulation/python` as the independent Python sub-project and Git repository for the MESTI port.
- Use `TASK_STATE.md` for compact active state and keep it under 100 lines.
- Use Git diffs, focused tests, and `git status --short` as the detailed handoff record.
- Treat `docs/legacy/mesti_python_port_plan_v7.md` as read-only historical context.
- Do not update the old `.codex/mesti_python_port_plan_v7.md` ledger during normal work.

## 2026-06-12 - cuDSS via nvmath-python in WSL

- Use WSL user `lky` and conda environment `optical_simulation` for real cuDSS checks.
- Prefer `nvmath.bindings.cudss` over a custom compiled extension when the binding is importable.
- Keep the exact command pattern and environment details in `docs/cudss_nvmath.md`.
- Keep Windows tests as the clean-skip and broad regression path until the WSL cumulative native `Bus error` is isolated.
