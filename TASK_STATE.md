# TASK_STATE

## Current goal
Continue the MESTI Julia-to-Python port v7 from Step 6 using this folder's Git repository and lightweight handoff files.

## Current status
- `Simulation/python` is the independent Python sub-project.
- Detailed v7 history is copied to `docs/legacy/mesti_python_port_plan_v7.md`.
- Treat the legacy v7 ledger as read-only historical context; do not update it during normal work.
- Added public README warning/provenance text for the planned `pyMESTI` upload.
- V7 Steps 0-5 are complete.
- Latest legacy full local suite result: 144 tests, 9 skipped.
- Legacy CodeGraph status after source edits: 11 files, 410 nodes, 1,192 edges.

## Completed v7 steps
- Step 0: created the v7 continuation plan.
- Step 1: inventoried Julia surfaces across `src/`, `examples/`, `MPI/`, and `mumps/`.
- Step 2: completed current-scope `mesti_subpixel_smoothing` rectangular `Cuboid` parity and explicit `Ball` unsupported stubs.
- Step 3: added raw-MUMPS public facade coverage with explicit unsupported raw invocation.
- Step 4: ported compact raw-MUMPS demo intent for `mumps/basic_solve.jl` and `mumps/schur_complement.jl`.
- Step 5: completed reachable option/convenience audit for `mesti`, `mesti2s`, and `mesti_matrix_solver`.

## Active next step
Start Step 6: translate standalone example helpers as reusable Python functions or explicit script-only compatibility stubs:
- `asp`
- `build_epsilon_disorder`
- `build_epsilon_disorder_3d`
- `plot_and_compare_distribution`

Prefer importable helpers when they improve exact Julia function coverage. Keep plotting-only helpers as documented script-only or explicit unsupported stubs if they would add heavy visual dependencies.

## Important files
- `mesti/examples.py`: likely target for reusable helper functions.
- `examples/`: runnable translated example scripts.
- `tests/`: Python test suite.
- `mesti/README.md`: API and compatibility documentation.
- `docs/legacy/mesti_python_port_plan_v7.md`: read-only historical context.

## Constraints
- Do not start production memory/speed optimization in v7.
- Do not rewrite unrelated modules.
- Do not update `.codex/mesti_python_port_plan_v7.md` during normal work.
- Use Git diffs, focused tests, and `TASK_STATE.md` as active handoff memory.
- Keep unsupported Julia paths explicit with clear errors or documented stubs.

## Verification
- README warning/provenance changes: verify with `Get-Content -TotalCount 20 README.md` and `Get-Content -TotalCount 20 mesti\README.md`.
- Documentation-only changes: run `git diff --stat` and `git status --short`.
- Source changes: run focused tests for touched surfaces; use `python -m unittest discover -s tests` before broad handoff when practical.
- After source edits, refresh CodeGraph from `mesti` with `codegraph sync` then `codegraph status`.

## Known blockers
- No current blocker for v7 translation work.
- Production-size `Ws300 Ls37.5` memory failure is deferred to v8 or later.
- MPI translation decision remains open for Step 7.
