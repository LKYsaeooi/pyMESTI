# Reusable MESTI Julia-to-Python Port Plan V7

Date started: 2026-06-05
Project root: `D:\BaiduSyncdisk\Projects\Q project`
V7 priority: complete translation of the Julia project surface first; defer
memory and speed optimization until the next phase unless needed to keep
correctness tests runnable.

V6 is complete. Treat `.codex/mesti_python_port_plan_v6.md` as the latest
completed historical baseline for production-solver diagnostics. Continue
active translation-completeness work from this v7 file only.

Project-local handoff files are allowed because the project root contains
`.codex-sync-enabled`, and the user explicitly requested
`.codex/mesti_python_port_plan_v7.md`.

## Mandatory Handoff Prompt

Paste this prompt at the start of every future chat working on v7:

```text
You are continuing the MESTI Python port v7 in
`D:\BaiduSyncdisk\Projects\Q project`.

Before making changes:
1. Read `.codex/mesti_python_port_plan_v7.md`.
2. Read the latest "Current Status Snapshot", "Julia-to-Python Inventory",
   "Step-by-Step V7 Roadmap", "Detailed Progress Log", "Open Decisions",
   "Blockers", "Next Step", and "Next Prompt" sections.
3. Inspect every file mentioned in the latest progress entry before editing.
4. Treat `.codex/mesti_python_port_plan_v6.md` as completed historical context.
   Read its final completion summary only if you need exact v6 solver or
   production-memory details.
5. If Julia reference data is needed, use WSL Julia through
   `wsl.exe --user lky -- bash -ic ...`; do not rely on Windows PATH.
6. Continue only from the recorded next step unless the user explicitly changes
   the priority.

While translating:
- Update `.codex/mesti_python_port_plan_v7.md` after every completed, failed,
  interrupted, or documentation-only step before giving the final response.
- Treat this v7 file as the authoritative progress ledger for the current
  phase. Save progress here even for exploratory, failed, or partially
  completed work.
- The progress update must be detailed enough for another chat with no context
  to continue. Record files changed, Julia source files used, Python
  functions/classes modified, comments added/revised/reviewed, fixtures added
  or regenerated, backend capability findings, indexing and dtype/layout
  decisions, unsupported-path decisions, commands run, summarized command
  outputs, exact test outcomes, numerical tolerances, memory/time metrics,
  mismatches, blockers, and exact next recommended prompt.
- Update the roadmap checkbox for every completed step. If a step is only
  partly done, leave it unchecked and add an "Interrupted" progress entry with
  exact partial state and risks.
- Keep unsupported Julia paths explicit with `NotImplementedError`,
  `UnsupportedMumpsOperation`, or clear validation errors until the active
  Python backend demonstrably honors them and they have tests, documentation,
  and recorded metrics.
- After any source-code step, refresh CodeGraph from
  `Simulation/python/mesti` and record the `codegraph sync` and/or
  `codegraph status` output in this file. For documentation-only steps, record
  whether CodeGraph was checked and why no source refresh was needed.
- Do not start memory/speed optimization work in v7 unless it is necessary to
  keep a small correctness test runnable. Production-size `Ws300 Ls37.5`
  optimization is explicitly deferred to v8 or later.
- Do not store secrets, credentials, tokens, or unrelated private
  machine-specific data.
- Do not overwrite unrelated user changes.
```

## Environment And Runtime

- Project root:
  `D:\BaiduSyncdisk\Projects\Q project`
- Python package:
  `Simulation/python/mesti`
- Python tests:
  `Simulation/python/tests`
- Original Julia package:
  `Simulation/julia/MESTI.jl-0.5.1`
- This project folder is not a Git repository. Direct file inspection and the
  `.codex/` handoff files are the source of truth.
- Julia is not expected to be on the Windows PATH. Use WSL user `lky` and an
  interactive shell so the user's Julia/MPI/MKL/MUMPS environment loads:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia --version'
```

- Windows/current-Python full Python suite:

```powershell
Set-Location 'D:\BaiduSyncdisk\Projects\Q project\Simulation\python'
python -m unittest discover -s tests
```

- Windows conda environment used in older plans:

```powershell
Set-Location 'D:\BaiduSyncdisk\Projects\Q project\Simulation\python'
conda run -n simu_scattering_light python -m unittest discover -s tests
```

- WSL base Python full suite:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project/Simulation/python" && python -m unittest discover -s tests'
```

- Nested package CodeGraph path:

```text
D:\BaiduSyncdisk\Projects\Q project\Simulation\python\mesti\.codegraph
```

- Preferred CodeGraph refresh command after source-code changes:

```powershell
Set-Location 'D:\BaiduSyncdisk\Projects\Q project\Simulation\python\mesti'
codegraph sync
codegraph status
```

If the shell command is unavailable, use the CodeGraph MCP tools with
`projectPath = "D:\\BaiduSyncdisk\\Projects\\Q project\\Simulation\\python\\mesti"`.

## Current Status Snapshot

- V6 production-solver slice is complete. It added explicit
  `Opts.use_single_precision_MUMPS=True` support for `mumpspy`, but the
  production-size `Ws300 Ls37.5` run remains blocked by memory and is not a v7
  target.
- V7 priority chosen by the user:
  translate every function and user-facing script surface of the Julia project
  first, then optimize memory and speed in a later phase.
- First v7 implementation slice is complete:
  added `Simulation/python/mesti/mumps.py`, a Python compatibility facade for
  Julia raw-MUMPS helper names without the Julia `!` suffix.
- The new raw-MUMPS facade provides SciPy/SuperLU or dense-NumPy backed small
  algebra helpers for `Mumps`, `mumps_solve`, `mumps_factorize`, `mumps_det`,
  `mumps_schur_complement`, and `mumps_select_inv`.
- The facade also exposes state/control/display/predicate helpers such as
  `set_icntl`, `set_cntl`, `set_keep`, `set_job`, `provide_matrix`,
  `provide_rhs`, `get_rhs`, `get_sol`, `get_schur_complement`,
  `display_icntl`, `display_cntl`, `display_keep`, `sparse_rhs`,
  `dense_rhs`, and related predicates.
- Raw C/MPI invocation is still explicitly unsupported and tested through
  `UnsupportedMumpsOperation`. This is intentional because Python does not
  reimplement Julia's pointer-level MUMPS object or MPI worker orchestration.
- Second v7 implementation slice is complete:
  added 2D TE inverse-epsilon subpixel smoothing for rectangular `Cuboid`
  domains/objects, fixture-backed against Julia `mesti_subpixel_smoothing`.
- `mesti_subpixel_smoothing` now supports TM-only, TE-only, and combined
  TM+TE 2D calls. TE-only returns `(inv_epsilon_yy, inv_epsilon_zz,
  inv_epsilon_yz)`, while combined TM+TE returns `(epsilon_xx,
  (inv_epsilon_yy, inv_epsilon_zz, inv_epsilon_yz))`, matching Julia's return
  structure.
- Third v7 implementation slice is complete:
  added a small 3D face-planar rectangular `Cuboid` tensor-output slice for
  `mesti_subpixel_smoothing`, fixture-backed against Julia. This covers all
  nine returned 3D tensor components for a slab-style Cuboid and
  `without_sb=True`.
- Fourth v7 implementation slice is complete:
  added an explicit `Ball` compatibility stub for curved GeometryPrimitives
  shapes used by the Julia examples. Python can construct 2D/3D `Ball` objects
  for API familiarity, but `mesti_subpixel_smoothing` raises a specific
  `NotImplementedError` for `Ball` domains or objects.
- Fifth v7 implementation slice is complete:
  replaced the old one-axis 3D local fill approximation with a Python port of
  Julia/GeometryPrimitives local-plane `volfrac` logic for rectangular
  `Cuboid` edge/corner cuts. The v7 3D Cuboid fixture now contains both the
  original face-planar slab prefix and a finite edge/corner Cuboid prefix.
- Sixth v7 implementation slice is complete:
  added compact Python translations of the user-facing intent in
  `mumps/basic_solve.jl` and `mumps/schur_complement.jl`. The new importable
  helpers and thin scripts exercise sparse multi-RHS solves, single/double
  precision input paths, Schur-complement extraction, one-based ICNTL controls,
  and Python zero-based Schur selectors while keeping raw C/MPI invocation
  explicitly unsupported.
- Seventh v7 implementation slice is complete:
  completed the Step 5 reachable option/convenience audit for direct `mesti`,
  `mesti2s`, and `mesti_matrix_solver`. Python now supports Julia-style
  positional `Opts` overloads for `mesti(syst, B, opts)` and
  `mesti(syst, B, C, opts)`, and `mesti2s` now rejects `opts.prefactor`
  explicitly instead of letting it leak into the delegated direct `mesti` call.
- Full local Python suite passed after the latest v7 slice:
  `144 tests`, `9 skipped`.
- CodeGraph was refreshed after source edits and is up to date:
  `11 files`, `410 nodes`, `1,192 edges`.

## Julia-to-Python Inventory

Status values:
`ported+tested`, `ported+under-tested`, `missing`,
`intentionally-delegated`, `explicit-unsupported-stub`.

| Julia source surface | Julia names | Python target | Current status | Notes / next action |
| --- | --- | --- | --- | --- |
| `src/mesti_main.jl` structs | `Source_struct`, `Syst` | `mesti/types.py` | `ported+tested` | Python uses zero-based `ind` and `pos`; keep this documented. |
| `src/mesti_main.jl` entry | `mesti` overload family | `mesti/mesti.py` | `ported+tested` | 2D TM and fixture-backed 3D direct paths covered; Julia-style positional `Opts` overloads are now tested. |
| `src/mesti2s.jl` selectors | `channel_type`, `channel_index`, `wavefront` | `mesti/types.py` | `ported+tested` | Python zero-based channel indices. |
| `src/mesti2s.jl` entry | `mesti2s` overload family | `mesti/mesti2s.py` | `ported+tested` | 2D TM and fixture-backed 3D diagonal/off-diagonal slices covered; positional `Opts` overload, symmetrized-K guards, and `opts.prefactor` rejection are tested. |
| `src/mesti_matrix_solver.jl` structs | `Matrices`, `Opts`, `Info` | `mesti/types.py` | `ported+tested` | Advanced MUMPS options mostly explicit validation errors. |
| `src/mesti_matrix_solver.jl` solver | `mesti_matrix_solver!` | `mesti/solver.py:mesti_matrix_solver` | `ported+tested` | Python name drops `!`; SciPy, `mumpspy`, and limited `python-mumps` support. |
| `src/mesti_matrix_solver.jl` internals | `JULIA_factorize`, `MUMPS_analyze_and_factorize`, `MUMPS_error_message` | `mesti/solver.py` | `intentionally-delegated` | Internal Julia implementation details; add explicit Python helpers only if a translated caller needs them. |
| `src/mesti_build_fdfd_matrix.jl` struct | `PML` | `mesti/types.py` | `ported+tested` | Direction and side fields are Python conveniences used by direct PML parsing. |
| `src/mesti_build_fdfd_matrix.jl` assembly | `mesti_build_fdfd_matrix` overloads | `mesti/fdfd_matrix.py` | `ported+tested` | 2D TM plus 3D diagonal/off-diagonal fixture coverage. |
| `src/mesti_build_fdfd_matrix.jl` helpers | `convert_BC`, `build_ddx_E`, `build_ave_x_Ex` | `mesti/boundary.py` | `ported+tested` | Keep private `check_BC_and_grid` behavior covered through public matrix tests. |
| `src/mesti_set_PML_params.jl` | `mesti_set_PML_params` | `mesti/boundary.py` | `ported+tested` | Existing boundary/PML tests cover default filling. |
| `src/mesti_build_transverse_function.jl` | `mesti_build_transverse_function`, `mesti_build_transverse_function_derivative`, `convert_BC_1d` | `mesti/channels.py` | `ported+tested` | 2D/3D channel tests cover this surface. |
| `src/mesti_setup_longitudinal.jl` | `Side`, `mesti_setup_longitudinal` | `mesti/types.py`, `mesti/channels.py` | `ported+tested` | Fixture and unit coverage exists through channel building. |
| `src/mesti_build_channels.jl` structs | `Channels_one_sided`, `Channels_two_sided` | `mesti/types.py` | `ported+tested` | Abstract `Channels` type is intentionally not represented. |
| `src/mesti_build_channels.jl` | `mesti_build_channels`, `convert_BC_to_transverse` | `mesti/channels.py`, `mesti/boundary.py` | `ported+tested` | Overload-style Python dispatcher exists. |
| `src/mesti_subpixel_smoothing.jl` | `mesti_subpixel_smoothing` | `mesti/subpixel.py` | `ported+under-tested` | 2D TM, 2D TE inverse-epsilon, and small 3D tensor rectangular `Cuboid` paths are fixture-backed, including edge/corner local-plane `volfrac` cuts. `Ball` curved-shape compatibility stubs are explicit unsupported paths; multiple-object and boundary-variant 3D cases remain candidates for future hardening. |
| `src/mumps3_struc.jl` | `Mumps` | `mesti/mumps.py` | `ported+tested` | Python facade stores matrix/RHS/factorization state; not a raw C pointer object. |
| `src/mumps3_interface.jl` raw call | `invoke_mumps!`, `invoke_mumps_unsafe!` | `mesti/mumps.py` | `explicit-unsupported-stub` | Raises `UnsupportedMumpsOperation`; test added. |
| `src/mumps3_interface.jl` controls | `set_keep!`, `set_icntl!`, `set_cntl!`, `set_job!`, `set_save_dir!`, `set_save_prefix!` | `mesti/mumps.py` | `ported+tested` | Python names drop `!`; indices are 1-based for ICNTL/CNTL/KEEP like Julia docs. |
| `src/mumps3_interface.jl` matrix/RHS | `provide_matrix!`, `provide_rhs!`, `provide_rhs_sparse!`, `provide_rhs_dense!`, `get_rhs!`, `get_rhs`, `get_sol!`, `get_sol` | `mesti/mumps.py` | `ported+tested` | Uses Python object state, not pointer loading. |
| `src/mumps3_interface.jl` Schur | `set_schur_centralized_by_column!`, `get_schur_complement!`, `get_schur_complement` | `mesti/mumps.py` | `ported+tested` | Uses dense Schur formula for small compatibility cases. |
| `src/mumps3_interface.jl` predicates | `is_matrix_assembled`, `is_matrix_distributed`, `is_rhs_dense`, `is_sol_central`, `has_det`, `is_symmetric`, `is_posdef`, `has_matrix`, `has_rhs`, `has_schur` | `mesti/mumps.py` | `ported+tested` | Basic state predicates tested; add edge tests only if future callers need exact MUMPS manual semantics. |
| `src/mumps3_convenience_wrappers.jl` high-level | `mumps_solve!`, `mumps_solve`, `mumps_factorize!`, `mumps_factorize`, `mumps_det!`, `mumps_det`, `mumps_schur_complement!`, `mumps_schur_complement`, `mumps_select_inv!`, `mumps_select_inv`, `initialize!`, `finalize!`, `finalize_unsafe!` | `mesti/mumps.py` | `ported+tested` | Python names drop `!`, using `_inplace` suffix for mutating helpers. Production raw-MUMPS parity is not claimed. |
| `src/mumps3_icntl_alibis.jl` | `default_icntl!`, stream/print helpers, matrix/RHS mode helpers, `toggle_null_pivot!` | `mesti/mumps.py` | `ported+tested` | `transpose!` is exposed as `transpose_mumps` because `transpose` is too generic in Python. |
| `src/mumps3_printing.jl` | `display_icntl`, `display_cntl`, `display_keep`, `Base.show` | `mesti/mumps.py` | `ported+tested` | Returns strings instead of printing to Julia IO objects. |
| `examples/2d_reflection_matrix_Gaussian_beams` | script workflow | `mesti/examples.py`, `Simulation/python/examples/reflection_matrix_gaussian_beams.py` | `ported+tested` | Reduced fixture-backed translation exists. |
| `examples/2d_open_channel_through_disorder` | main script | `Simulation/python/examples/open_channel_through_disorder.py` | `ported+tested` | Reduced fixture-backed translation exists. |
| `examples/2d_open_channel_through_disorder` helpers | `build_epsilon_disorder`, `plot_and_compare_distribution` | example scripts / not importable helper yet | `ported+under-tested` | Next v7 example target: expose reusable Python helper(s) or explicit documented script-only status. |
| `examples/2d_focusing_inside_disorder_with_phase_conjugation` | main script, `build_epsilon_disorder` | `Simulation/python/examples/focusing_inside_disorder_with_phase_conjugation.py` | `ported+tested` | Reduced fixture-backed translation exists; helper not yet standalone importable. |
| `examples/2d_metalens_focusing_via_angular_spectrum_propagation` | main script, `asp` | `Simulation/python/examples/metalens_focusing_via_angular_spectrum_propagation.py` | `ported+tested` | Reduced fixture-backed translation exists; next v7 target should expose/test `asp` as a helper if exact Julia function coverage is required. |
| `examples/3d_open_channel_through_disorder` | main script, `build_epsilon_disorder_3d`, `plot_and_compare_distribution` | `Simulation/python/examples/open_channel_through_disorder_3d.py` | `ported+tested` | Reduced fixture-backed translation exists; helper functions not yet separately inventoried/testable. |
| `MPI/hybrid_mpi.jl` | MPI worker example | none | `missing` | Next MPI target: translate as a documented Python MPI skeleton or explicit unsupported script. Do not optimize production memory in v7. |
| `mumps/basic_solve.jl`, `mumps/schur_complement.jl` | raw MUMPS demos | `mesti/examples.py`, `Simulation/python/examples/basic_mumps_solve.py`, `Simulation/python/examples/mumps_schur_complement.py`, `mesti/mumps.py` plus tests | `ported+tested` | Compact deterministic Python demos cover the Julia scripts' user-facing solve and Schur-complement intent. They do not claim raw C/MPI invocation parity. |
| notebooks / Makefiles | `.ipynb`, OS-specific `Makefile.inc` | documentation only | `intentionally-delegated` | Out of v7 scope unless unique executable logic is found. |

## Step-by-Step V7 Roadmap

- [x] Step 0. Create this v7 plan file with continuation prompt, runtime notes,
  inventory statuses, roadmap, and progress logging rules.
- [x] Step 1. Build an initial Julia-to-Python inventory covering `src/`,
  `examples/`, `MPI/`, and `mumps/`.
- [x] Step 2. Finish missing subpixel-smoothing translation: remaining 3D
  edge/corner tensor smoothing and curved GeometryPrimitives-equivalent shapes
  where Julia supports them. Add Julia-generated fixtures and tests.
- [x] Step 3. Add the first raw-MUMPS public facade slice so Julia MUMPS helper
  names have Python-callable coverage or explicit unsupported errors.
- [x] Step 4. Deepen raw-MUMPS facade coverage only where needed: add parity
  fixtures for `mumps/basic_solve.jl` and `mumps/schur_complement.jl`, and add
  demo scripts if they are considered user-facing.
- [x] Step 5. Complete `mesti`, `mesti2s`, and `mesti_matrix_solver` option
  audit for any remaining overload-style Julia convenience forms or currently
  implicit unsupported paths.
- [ ] Step 6. Translate standalone example helpers as reusable Python functions
  or explicit script-only compatibility stubs: `asp`, `build_epsilon_disorder`,
  `build_epsilon_disorder_3d`, and `plot_and_compare_distribution`.
- [ ] Step 7. Add an MPI translation decision for `MPI/hybrid_mpi.jl`: either a
  runnable `mpi4py` skeleton matching the Julia script's user-facing flow or a
  tested explicit unsupported script with migration notes.
- [ ] Step 8. Update docs/API map so every inventory row is `ported+tested`,
  `intentionally-delegated`, or `explicit-unsupported-stub`; no row should stay
  `missing` before v7 closes.
- [ ] Step 9. V7 closeout: run focused tests, full Windows/current-Python
  suite, WSL suite when practical, refresh CodeGraph, and write a final
  completion summary. Then open v8 for memory/speed optimization.

## Detailed Progress Log

### 2026-06-06 - Completed reachable option/convenience audit

Intent:
Perform Step 5 while honoring the user's token-saving request. Use CodeGraph
for the Python surface, then targeted `rg`/small source excerpts for Julia
option and overload behavior instead of re-reading every file named by the
latest progress entry.

Context inspected:
- Used CodeGraph first with projectPath
  `D:\BaiduSyncdisk\Projects\Q project\Simulation\python\mesti` to inspect
  `mesti`, `mesti2s`, `mesti_matrix_solver`, `Opts`, `Syst`, and `Matrices`.
- Targeted Julia inspection:
  `src/mesti_main.jl` overload declarations and option checks,
  `src/mesti2s.jl` overload declarations and `opts.prefactor` guard,
  and `src/mesti_matrix_solver.jl` option validation.
- Targeted Python inspection:
  `Simulation/python/mesti/mesti.py`,
  `Simulation/python/mesti/mesti2s.py`,
  `Simulation/python/mesti/solver.py`,
  `Simulation/python/mesti/types.py`,
  `Simulation/python/tests/test_mesti.py`,
  `Simulation/python/tests/test_mesti2s.py`, and
  `Simulation/python/tests/test_solver.py`.

Python files changed:
- Updated `Simulation/python/mesti/mesti.py`.
- Updated `Simulation/python/mesti/mesti2s.py`.
- Updated `Simulation/python/tests/test_mesti.py`.
- Updated `Simulation/python/tests/test_mesti2s.py`.
- Updated `Simulation/python/mesti/README.md`.
- Updated this handoff file:
  `.codex/mesti_python_port_plan_v7.md`.

Implementation details:
- Added direct `mesti` positional-`Opts` overload normalization:
  - `mesti(syst, B, opts)` now maps to the field-profile keyword form.
  - `mesti(syst, B, C, opts)` now maps to the projected keyword form.
  - Duplicate positional/keyword `opts` or combining `mesti(syst, B, opts)`
    with `D` now raises `TypeError`.
- Added `mesti2s` validation for `opts.prefactor`.
  Julia rejects this option because `mesti2s` applies the `-2i` scattering
  prefactor internally; Python now raises a clear `ValueError` instead of
  passing the value into the delegated direct `mesti` call.
- Confirmed `mesti2s` already had the positional `Opts` convenience form.
- Confirmed `mesti_matrix_solver` already has explicit validation/tests for
  reachable unsupported options including `symmetrize_K`, `analysis_only`,
  `store_ordering`, `use_given_ordering`, `ordering`,
  `iterative_refinement`, BLR controls, `nthreads_OMP`, `use_L0_threads`,
  `use_METIS`, `write_LU_factor_to_disk`, and backend-scoped
  `use_single_precision_MUMPS`.
- Updated README conventions to document the direct `mesti` options overloads
  and `mesti2s` prefactor rejection.

Tests added/updated:
- Added
  `test_positional_opts_overload_for_field_profile_matches_keyword_opts`.
- Added `test_positional_opts_overload_for_projection_matches_keyword_opts`.
- Added `test_prefactor_option_is_rejected_for_mesti2s`.

Commands run and outcomes:
- `python -m unittest discover -s tests -p test_mesti.py`
  - Passed after the first direct-`mesti` edit: `21 tests`, `0 failures`.
- `python -m unittest discover -s tests`
  - Passed after the first direct-`mesti` edit: `143 tests`, `9 skipped`.
- `codegraph sync` from `Simulation/python/mesti`
  - First sync after `mesti.py`: synced `1 changed files`; modified `1`,
    `43 nodes` in `160ms`.
- `codegraph status` from `Simulation/python/mesti`
  - First status after `mesti.py`: `11 files`, `410 nodes`, `1,181 edges`,
    DB size `1.04 MB`, backend `node:sqlite - built-in (full WAL)`,
    `[OK] Index is up to date`.
- `python -m unittest discover -s tests -p test_mesti.py`
  - Passed after the `mesti2s` prefactor edit: `21 tests`, `0 failures`.
- `python -m unittest discover -s tests -p test_mesti2s.py`
  - Passed after the `mesti2s` prefactor edit: `31 tests`, `1 skipped`.
- `python -m unittest discover -s tests`
  - Passed after all Step 5 edits: `144 tests`, `9 skipped`.
- `codegraph sync` from `Simulation/python/mesti`
  - Final sync after `mesti2s.py`: synced `1 changed files`; modified `1`,
    `85 nodes` in `193ms`.
- `codegraph status` from `Simulation/python/mesti`
  - Final status: `11 files`, `410 nodes`, `1,192 edges`, DB size `1.13 MB`,
    backend `node:sqlite - built-in (full WAL)`,
    `[OK] Index is up to date`.

Known limitations from this slice:
- This was an audit/fix pass for reachable option/convenience behavior, not a
  new numerical fixture slice.
- Advanced MUMPS options remain explicit unsupported paths unless a verified
  Python backend supports them.
- No production memory/speed optimization was attempted.

Next exact action:
Start Step 6. Translate standalone example helpers as reusable Python
functions or explicit script-only compatibility stubs: `asp`,
`build_epsilon_disorder`, `build_epsilon_disorder_3d`, and
`plot_and_compare_distribution`. Prefer importable helper functions when they
improve exact Julia function coverage, but keep plotting-only helpers as
documented script-only or explicit unsupported stubs if they would add heavy
visual dependencies.

### 2026-06-06 - Added raw-MUMPS demo translations

Intent:
Perform Step 4 by inspecting Julia's raw-MUMPS demo scripts and adding only the
user-facing Python parity surface that improves translation completeness. Keep
raw C/MPI invocation explicitly unsupported and do not start production
memory/speed optimization.

Context inspected:
- Used CodeGraph first with projectPath
  `D:\BaiduSyncdisk\Projects\Q project\Simulation\python\mesti` to inspect the
  existing `mesti/mumps.py` facade, package exports, README, tests, and example
  patterns.
- Read `Simulation/julia/MESTI.jl-0.5.1/mumps/basic_solve.jl`.
  The Julia script runs random sparse multi-RHS solves for double and single
  precision through `mumps_solve`.
- Read `Simulation/julia/MESTI.jl-0.5.1/mumps/schur_complement.jl`.
  The Julia script builds an augmented block matrix, configures raw MUMPS jobs
  and Schur indices, then validates `D - C * inv(A) * B`.
- Read `Simulation/python/mesti/mumps.py`,
  `Simulation/python/tests/test_mumps_compat.py`,
  `Simulation/python/mesti/examples.py`,
  `Simulation/python/mesti/__init__.py`,
  `Simulation/python/mesti/README.md`, and existing example/test patterns.

Python files changed:
- Updated `Simulation/python/mesti/examples.py`.
- Updated `Simulation/python/mesti/__init__.py`.
- Added `Simulation/python/examples/basic_mumps_solve.py`.
- Added `Simulation/python/examples/mumps_schur_complement.py`.
- Updated `Simulation/python/tests/test_mumps_compat.py`.
- Updated `Simulation/python/mesti/README.md`.
- Updated this handoff file:
  `.codex/mesti_python_port_plan_v7.md`.

Implementation details:
- Added `BasicMumpsSolveDemoResult` and `basic_mumps_solve_demo`.
  This deterministic helper mirrors the useful public contract of
  `mumps/basic_solve.jl` with a tiny sparse tridiagonal system, multiple RHS
  columns, dense or sparse RHS input, and `complex128`/`complex64` input modes.
- Added `MumpsSchurComplementDemoResult` and
  `mumps_schur_complement_demo`. This helper mirrors the useful public
  contract of `mumps/schur_complement.jl` with a tiny deterministic block
  matrix and compares the compatibility facade output with
  `D - C @ solve(A, B)`.
- The Schur demo deliberately uses Python zero-based Schur selectors
  `np.arange(m, m + n)` for the trailing block that Julia selects with
  one-based `m+1:m+n`.
- The Schur demo exercises state/control surface calls:
  `Mumps`, `set_icntl`, `set_job`, `set_schur_centralized_by_column`,
  `mumps_schur_complement_inplace`, and `get_schur_complement`.
- Raw `invoke_mumps` remains an explicit `UnsupportedMumpsOperation`; this
  slice did not simulate Julia raw analysis/factorization jobs or MPI workers.
- Added thin runnable scripts:
  `examples/basic_mumps_solve.py` and `examples/mumps_schur_complement.py`.
  They call the importable helpers and print compact residual/error summaries.
- Exported the new result dataclasses and helpers from `mesti.__init__`.
- Updated README overview, validation status, package map, quick-start
  commands, low-level MUMPS notes, and known gaps.

Tests added/updated:
- Added `test_basic_mumps_solve_demo_matches_julia_demo_contract`.
- Added `test_schur_complement_demo_matches_julia_demo_contract`.
- Added `test_raw_mumps_demo_scripts_are_importable`.
- The tests cover both `complex128` and `complex64` input modes. The
  compatibility facade internally computes in `complex128`, so the single
  precision path is an input-surface parity check rather than raw MUMPS
  single-precision backend parity.

Commands run and outcomes:
- `python -m unittest discover -s tests -p test_mumps_compat.py`
  - Passed: `11 tests`, `0 failures`.
- `python -m unittest discover -s tests`
  - Passed: `141 tests`, `9 skipped`.
- `python examples\basic_mumps_solve.py`
  - Passed.
  - Printed `complex128 sparse_rhs=False: residual=2.759e-16, tolerance=3.733e-08`.
  - Printed `complex64 sparse_rhs=True: residual=3.727e-16, tolerance=8.649e-04`.
- `python examples\mumps_schur_complement.py`
  - Passed.
  - Printed `complex128: relative_error=0.000e+00, tolerance=2.980e-08, indices=[6, 7, 8]`.
  - Printed `complex64: relative_error=0.000e+00, tolerance=6.905e-04, indices=[6, 7, 8]`.
- `codegraph sync` from `Simulation/python/mesti`
  - Synced `2 changed files`.
  - Modified `2`, `31 nodes` in `189ms`.
- `codegraph status` from `Simulation/python/mesti`
  - Project: `D:\BaiduSyncdisk\Projects\Q project\Simulation\python\mesti`
  - Files: `11`
  - Nodes: `410`
  - Edges: `1,169`
  - DB size: `1.04 MB`
  - Backend: `node:sqlite - built-in (full WAL)`
  - Status: `[OK] Index is up to date`

Known limitations from this slice:
- No Julia `.mat` fixture was added for `mumps/basic_solve.jl` or
  `mumps/schur_complement.jl` because the scripts use random stress loops and
  raw MUMPS/MPI jobs. The Python replacements are deterministic parity demos
  for the user-facing algebra contract.
- The raw MUMPS C/MPI invocation path is still intentionally unsupported.
- The helper demos are tiny correctness checks, not production raw-MUMPS or
  production memory/speed benchmarks.

Next exact action:
Start Step 5. Audit remaining `mesti`, `mesti2s`, and `mesti_matrix_solver`
option/convenience surfaces against the Julia sources for overload-style forms
or implicit unsupported paths. Keep unsupported behavior explicit, add focused
tests/docs only where a Python caller can reach the surface, and do not start
production memory/speed optimization.

### 2026-06-06 - Completed 3D Cuboid edge/corner subpixel parity

Intent:
Finish the remaining Step 2 depth item by porting rectangular-Cuboid
edge/corner `surfpt_nearby`/`volfrac` behavior for small 3D subpixel
smoothing. Keep the work correctness-first and do not start memory/speed
optimization.

Context inspected:
- Read only targeted portions of `.codex/mesti_python_port_plan_v7.md` per the
  user's token-saving request.
- Used CodeGraph first for the MESTI Python subpixel surface. CodeGraph pointed
  to `mesti/subpixel.py` helpers including `_local_planar_fill_fraction`,
  `_nearest_cuboid_face`, `_smooth_2d_tm_cuboids`, and
  `mesti_subpixel_smoothing`.
- Used WSL Julia to probe `GeometryPrimitives.surfpt_nearby` and `volfrac`
  values for a finite Cuboid, confirming Julia's local normal and volume
  fractions at face, edge, and corner sample points.
- Inspected the installed Julia `GeometryPrimitives` `vxlcut.jl` volume-cut
  algorithm through WSL. The Python port mirrors the relevant `corner_bits`,
  `rvol_quadsect`, `rvol_gensect`, and `volfrac` branches for rectangular
  Cuboid local planes.

Files changed:
- Updated `Simulation/python/mesti/subpixel.py`.
- Updated `Simulation/python/tests/test_subpixel.py`.
- Updated `Simulation/python/tests/fixtures/generate_subpixel_3d_cuboid_v7_fixture.jl`.
- Regenerated `Simulation/python/tests/fixtures/subpixel_3d_cuboid_v7.mat`.
- Updated `Simulation/python/mesti/README.md`.
- Updated `Simulation/python/tests/fixtures/README.md`.
- Updated this handoff file:
  `.codex/mesti_python_port_plan_v7.md`.

Implementation details:
- Added `_cuboid_surface_point_and_normal` so outside points use the nearest
  clamped Cuboid surface and inside points use the nearest face, matching the
  Julia `surfpt_nearby` convention needed by the local-plane cut.
- Added Python translations of the GeometryPrimitives voxel-plane helpers:
  `_corner_bits`, `_is_quadsect`, `_edge_dir_quadsect`,
  `_relative_volume_quadsect`, `_relative_volume_gensect`, and
  `_plane_volume_fraction_3d`.
- Replaced `_local_planar_fill_fraction`'s old one-axis fill approximation with
  local-plane 2D/3D volume fractions. The 2D path embeds the pixel into a 3D
  unit-thickness voxel with zero z-normal, matching Julia's 2D `volfrac` route.
- Updated `_cuboid_surface_normal` to reuse the new surface-point/normal helper.
- Extended the v7 3D fixture with `edge3d_*` keys using:
  domain center `[1.0, 1.0, 1.0]`, domain widths `[2.0, 2.0, 2.0]`,
  object center `[0.75, 0.75, 0.75]`, object widths `[1.0, 1.0, 1.0]`,
  scalar epsilon `4.0`, and periodic boundaries in x/y/z.
- Kept the original slab-style `rect3d_*` fixture keys for the face-planar
  regression.
- Added a shared Python test helper for comparing all nine 3D tensor components
  for any fixture prefix.

Tests added/updated:
- Added `test_3d_cuboid_edge_corner_tensor_smoothing_matches_julia_fixture`.
- Added
  `test_3d_cuboid_edge_corner_without_subpixel_smoothing_matches_julia_fixture`.
- Updated the original 3D Cuboid tests to use the shared tensor comparison
  helper.
- Updated README/fixture docs so edge/corner Cuboid parity is no longer listed
  as future work. Curved `Ball` smoothing remains an explicit unsupported path.

Commands run and outcomes:
- `python -m py_compile Simulation/python/mesti/subpixel.py Simulation/python/tests/test_subpixel.py`
  - Passed.
- `python -m unittest discover -s tests -p test_subpixel.py`
  - Passed before fixture expansion: `11 tests`, `0 failures`.
- `wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_subpixel_3d_cuboid_v7_fixture.jl'`
  - Passed; regenerated `subpixel_3d_cuboid_v7.mat`.
  - Julia reported face-planar and finite edge/corner component size `(2, 2, 2)`.
  - WSL printed the usual oneAPI initialization banner and trailing
    localhost/NAT warning; fixture generation itself completed successfully.
- `python -m unittest discover -s tests -p test_subpixel.py`
  - Passed after fixture expansion: `13 tests`, `0 failures`.
- `python -m unittest discover -s tests`
  - Passed: `138 tests`, `9 skipped`.
- `codegraph sync` from `Simulation/python/mesti`
  - Synced `1 changed files`.
  - Modified `1`, `47 nodes` in `171ms`.
- `codegraph status` from `Simulation/python/mesti`
  - Project: `D:\BaiduSyncdisk\Projects\Q project\Simulation\python\mesti`
  - Files: `11`
  - Nodes: `402`
  - Edges: `1,136`
  - DB size: `1.04 MB`
  - Backend: `node:sqlite - built-in (full WAL)`
  - Status: `[OK] Index is up to date`
- `git status --short`
  - Failed because `D:\BaiduSyncdisk\Projects\Q project` is not a Git
    repository. This is expected; direct file inspection plus `.codex/` handoff
    files remain the continuity mechanism.

Known limitations from this slice:
- Curved `Ball` subpixel smoothing is still intentionally unsupported in
  Python and raises `NotImplementedError`.
- The new edge/corner fixture covers a small periodic finite Cuboid. Additional
  3D multiple-object and boundary-variant fixtures could harden this path later
  if needed.
- No memory or speed optimization was attempted.

Next exact action:
Move to Step 4. Deepen raw-MUMPS facade coverage only where it improves
translation-surface completeness: inspect `mumps/basic_solve.jl` and
`mumps/schur_complement.jl`, then add tiny parity fixtures and/or Python demo
scripts if those demos are considered user-facing. Keep raw C/MPI invocation
explicitly unsupported unless the user asks for runnable MPI/MUMPS bindings.

### 2026-06-05 - Added explicit Ball curved-shape unsupported stubs

Intent:
Continue Step 2 with the smaller remaining subpixel-completeness choice:
explicit curved-shape compatibility stubs/tests/docs. The Julia examples use
`GeometryPrimitives.Ball` for circular/cylindrical/spherical scatterers, while
Python still relies on Julia-generated fixtures or precomputed permittivity
arrays for curved-shape smoothing.

Context inspected:
- Read only targeted portions of `.codex/mesti_python_port_plan_v7.md` per the
  user's token-saving request.
- Used CodeGraph first for the MESTI Python translation surface.
- Searched the Julia/examples tree for GeometryPrimitives constructors. The
  project examples and fixture generators use `Ball` plus `Cuboid`; no other
  curved constructor appeared in the current search.

Python files changed:
- Updated `Simulation/python/mesti/subpixel.py`.
- Updated `Simulation/python/mesti/__init__.py`.
- Updated `Simulation/python/tests/test_subpixel.py`.
- Updated `Simulation/python/mesti/README.md`.
- Updated `Simulation/python/tests/fixtures/README.md`.
- Updated this handoff file:
  `.codex/mesti_python_port_plan_v7.md`.

Implementation details:
- Added a `Ball` dataclass compatibility stub with `center`, `radius`, and
  `ndim`.
- `Ball` validates:
  - center is one-dimensional;
  - dimension is 2 or 3;
  - radius is positive.
- Added `_unsupported_curved_shape_error` and `_validate_cuboid_objects`.
- `mesti_subpixel_smoothing` now raises a specific `NotImplementedError` for
  `Ball` used as either the domain or an object, with a message directing users
  to Julia-generated fixtures or precomputed epsilon arrays.
- Exported `Ball` from `mesti.__init__` and added it to `__all__`.
- Existing `Cuboid` behavior and fixtures were not changed.

Tests added/updated:
- Added `test_ball_compatibility_stub_validates_shape`.
- Added `test_curved_shape_subpixel_paths_are_explicit_unsupported`.
- Curved-shape tests cover:
  - construction of a 2D `Ball`;
  - invalid nonpositive radius;
  - invalid dimension;
  - 2D `Ball` object rejection;
  - `Ball` domain rejection;
  - 3D `Ball` object rejection with Julia-style 3D positional boundaries.

Commands run and outcomes:
- `python -m unittest discover -s tests -p test_subpixel.py`
  - Passed: `11 tests`, `0 failures`.
- `python -m py_compile Simulation/python/mesti/subpixel.py Simulation/python/mesti/__init__.py Simulation/python/tests/test_subpixel.py`
  - Passed.
- `python -m unittest discover -s tests`
  - Passed: `136 tests`, `9 skipped`.
- `codegraph sync` from `Simulation/python/mesti`
  - Synced `2 changed files`.
  - Modified `2`, `52 nodes` in `153ms`.
- `codegraph status` from `Simulation/python/mesti`
  - Project: `D:\BaiduSyncdisk\Projects\Q project\Simulation\python\mesti`
  - Files: `11`
  - Nodes: `395`
  - Edges: `1,120`
  - DB size: `1.04 MB`
  - Backend: `node:sqlite - built-in (full WAL)`
  - Status: `[OK] Index is up to date`

Known limitations from this slice:
- `Ball` is not a smoothing implementation; it is an explicit unsupported
  compatibility stub.
- Curved-shape smoothing still requires Julia-generated fixtures or precomputed
  arrays.
- No 3D edge/corner `surfpt_nearby`/`volfrac` parity was attempted.
- No memory or speed optimization was attempted.

Next exact action:
Continue Step 2 only if deeper subpixel completeness is still desired in v7:
implement 3D edge/corner local-plane `surfpt_nearby`/`volfrac` parity for
rectangular Cuboids with a small Julia fixture. Otherwise, if the user prefers
translation-surface breadth over deeper subpixel math, move to Step 4 or Step
5 as recorded in the roadmap.

### 2026-06-05 - Added small 3D face-planar Cuboid subpixel smoothing

Intent:
Continue Step 2 after the 2D TE slice by adding the next fixture-backed 3D
subpixel-smoothing surface. Keep the implementation scoped to rectangular
`Cuboid` geometry and do not start production memory/speed optimization.

Julia sources inspected:
- `Simulation/julia/MESTI.jl-0.5.1/src/mesti_subpixel_smoothing.jl`
  - 3D overload validation, domain normalization, periodic image handling, and
    Eo/Ex/Ey/Ez coordinate definitions.
  - 3D Eo/Ex/Ey/Ez Kottke smoothing loops.
  - `pick_epsilon_3d` component selection and boundary trimming.

Python files changed:
- Updated `Simulation/python/mesti/subpixel.py`.
- Updated `Simulation/python/tests/test_subpixel.py`.
- Added `Simulation/python/tests/fixtures/generate_subpixel_3d_cuboid_v7_fixture.jl`.
- Added/regenerated `Simulation/python/tests/fixtures/subpixel_3d_cuboid_v7.mat`.
- Updated `Simulation/python/mesti/README.md`.
- Updated `Simulation/python/tests/fixtures/README.md`.
- Updated this handoff file:
  `.codex/mesti_python_port_plan_v7.md`.

Implementation details:
- Generalized periodic-image handling to work for either 2D `(yBC, zBC)` or
  3D `(xBC, yBC, zBC)` boundaries.
- Generalized Cuboid corner sampling from hard-coded 2D offsets to
  N-dimensional offsets.
- Generalized nearest-face lookup for 2D/3D Cuboids.
- Added `_pick_epsilon_3d`, returning Julia's nine 3D components in order:
  `epsilon_xx`, `epsilon_xy`, `epsilon_xz`, `epsilon_yx`, `epsilon_yy`,
  `epsilon_yz`, `epsilon_zx`, `epsilon_zy`, `epsilon_zz`.
- Added `_smooth_epsilon_site` and `_smooth_3d_cuboids`.
  The 3D path initializes Eo/Ex/Ey/Ez tensor sites, applies the Kottke helper
  at partially occupied face-planar pixels, and then calls `_pick_epsilon_3d`.
- Extended `mesti_subpixel_smoothing` overload behavior:
  - 2D keeps the existing Python call shape:
    `(..., yBC, zBC, use_2D_TM=True, use_2D_TE=False, without_sb=False)`.
  - 3D uses Julia-style positional overload interpretation:
    `(..., xBC, yBC, zBC, without_sb=False)`.
    Because the Python signature already had 2D flags, the seventh positional
    slot (`use_2D_TM`) must be a string in 3D and is treated as `zBC`.
    The optional ninth positional slot (`use_2D_TE`) is accepted as 3D
    `without_sb` only when boolean; keyword `without_sb=True` also works.
- Updated validation so accidental mixed 2D/3D calls fail loudly.

Fixture details:
- Added `generate_subpixel_3d_cuboid_v7_fixture.jl`.
- Initial exploratory fixture used a finite 3D Cuboid with edge/corner partial
  pixels. Python mismatched Julia in the smoothed case because Julia's
  `surfpt_nearby`/`volfrac` handles edge/corner local-plane cuts, while this
  Python slice only implements the face-planar reduction used by the earlier
  2D support.
- The committed fixture was narrowed to a slab-style rectangular Cuboid:
  center `[0.75, 1.0, 1.0]`, widths `[1.0, 4.0, 4.0]`, domain center
  `[1.0, 1.0, 1.0]`, domain widths `[2.0, 2.0, 2.0]`, all boundaries
  `"periodic"`. This exercises 3D tensor smoothing with face-planar partial
  pixels and equal component shapes.
- Julia fixture regeneration reported component size `(2, 2, 2)` and wrote
  `Simulation/python/tests/fixtures/subpixel_3d_cuboid_v7.mat`.

Tests added/updated:
- Added `test_3d_cuboid_tensor_smoothing_matches_julia_fixture`.
- Added `test_3d_cuboid_without_subpixel_smoothing_matches_julia_fixture`.
- Updated unsupported-path coverage so a 3D Cuboid with missing `zBC` now
  raises a clear `TypeError` instead of the old 3D unsupported error.

Commands run and outcomes:
- `python -m py_compile Simulation/python/mesti/subpixel.py Simulation/python/tests/test_subpixel.py`
  - Passed before fixture generation.
- `wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_subpixel_3d_cuboid_v7_fixture.jl'`
  - Passed for the initial exploratory edge/corner fixture and wrote
    `subpixel_3d_cuboid_v7.mat`.
- `python -m unittest discover -s tests -p test_subpixel.py`
  - Failed in `test_3d_cuboid_tensor_smoothing_matches_julia_fixture`.
  - Component diagnostics showed mismatches across smoothed 3D tensor
    components, with `epsilon_xx` max absolute difference `0.375`, confirming
    the edge/corner local-plane gap.
- Regenerated the fixture after narrowing it to the face-planar slab Cuboid:
  `wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_subpixel_3d_cuboid_v7_fixture.jl'`
  - Passed; component size `(2, 2, 2)`.
  - WSL printed the usual oneAPI initialization banner and trailing
    localhost/NAT warning; fixture generation itself completed successfully.
- `python -m unittest discover -s tests -p test_subpixel.py`
  - Passed: `9 tests`, `0 failures`.
- `python -m unittest discover -s tests`
  - Passed: `134 tests`, `9 skipped`.
- `codegraph sync` from `Simulation/python/mesti`
  - Synced `1 changed files`.
  - Modified `1`, `35 nodes` in `156ms`.
- `codegraph status` from `Simulation/python/mesti`
  - Project: `D:\BaiduSyncdisk\Projects\Q project\Simulation\python\mesti`
  - Files: `11`
  - Nodes: `390`
  - Edges: `1,111`
  - DB size: `1.04 MB`
  - Backend: `node:sqlite - built-in (full WAL)`
  - Status: `[OK] Index is up to date`

Known limitations from this slice:
- The 3D fixture is a face-planar slab case. Full 3D edge/corner
  `surfpt_nearby`/`volfrac` parity is not implemented.
- Curved `GeometryPrimitives` shapes remain unsupported in Python.
- 3D boundary trimming is translated in `_pick_epsilon_3d`, but the new fixture
  uses periodic boundaries only. Add boundary-specific 3D subpixel fixtures if
  future callers need that surface.
- No memory or speed optimization was attempted.

Next exact action:
Continue Step 2. The remaining subpixel choices are now:
1. add explicit unsupported curved-shape compatibility stubs/tests and docs, or
2. implement 3D edge/corner local-plane `surfpt_nearby`/`volfrac` parity for
   rectangular Cuboids with a small Julia fixture.

### 2026-06-05 - Added 2D TE rectangular-Cuboid subpixel smoothing

Intent:
Continue Step 2 with the smallest fixture-backed subpixel slice after the
completed TM path: 2D TE inverse-epsilon outputs for rectangular
`GeometryPrimitives.Cuboid` objects. Keep 3D tensor smoothing, curved shapes,
and production memory/speed optimization out of this slice.

Julia sources inspected:
- `Simulation/julia/MESTI.jl-0.5.1/src/mesti_subpixel_smoothing.jl`
  - 2D overload validation and domain translation.
  - 2D TE `inv_epsilon_Eo_site`, `inv_epsilon_Ey_site`, and
    `inv_epsilon_Ez_site` assembly loops.
  - `pick_inv_epsilon_2d_TE`, `tau_trans`, `tau_inverse_trans`, and
    `Kottke_smoothing`.

Python files changed:
- Updated `Simulation/python/mesti/subpixel.py`.
- Updated `Simulation/python/tests/test_subpixel.py`.
- Updated `Simulation/python/tests/fixtures/generate_subpixel_2d_tm_v5_fixture.jl`.
- Regenerated `Simulation/python/tests/fixtures/subpixel_2d_tm_v5.mat`.
- Updated `Simulation/python/mesti/README.md`.
- Updated `Simulation/python/tests/fixtures/README.md`.
- Updated this handoff file:
  `.codex/mesti_python_port_plan_v7.md`.

Implementation details:
- Added `_pick_inv_epsilon_2d_te` matching Julia boundary trimming:
  `inv_epsilon_yy` from Ey sites, `inv_epsilon_zz` from Ez sites, and
  `inv_epsilon_yz` from Eo sites.
- Added deterministic Python translations of Julia's Kottke tensor helpers:
  `_tau_trans`, `_tau_inverse_trans`, `_orthonormal_basis_from_normal`, and
  `_kottke_smoothing`.
- Added 2D rectangular-Cuboid TE assembly via `_smooth_2d_te_cuboids`.
  Coordinates mirror Julia's Yee-site layout:
  Eo `(Ez_y, Ey_z)`, Ey `(Ey_y, Ey_z)`, and Ez `(Ez_y, Ez_z)`.
- TE smoothing stores inverse-epsilon tensors. For partially occupied pixels,
  it inverts the current inverse tensor, applies Kottke smoothing against the
  object scalar permittivity tensor, then stores the inverse of the smoothed
  tensor, matching Julia's `inv(Kottke_smoothing(..., inv(inv_epsilon_site)))`
  pattern.
- `mesti_subpixel_smoothing` now supports:
  - TM-only: returns `epsilon_xx`.
  - TE-only with `use_2D_TM=False, use_2D_TE=True`: returns
    `(inv_epsilon_yy, inv_epsilon_zz, inv_epsilon_yz)`.
  - Combined TM+TE: returns `(epsilon_xx,
    (inv_epsilon_yy, inv_epsilon_zz, inv_epsilon_yz))`.
- Removed the previous explicit `NotImplementedError` for 2D TE. The remaining
  explicit unsupported paths are non-2D domains and non-`Cuboid` objects.
- Kept Python zero-based/user-facing conventions unchanged. ICNTL/CNTL/MUMPS
  conventions were not touched.

Fixture details:
- Extended `generate_subpixel_2d_tm_v5_fixture.jl` to record rectangular TE
  arrays:
  `rect_inv_epsilon_yy`, `rect_inv_epsilon_zz`, `rect_inv_epsilon_yz`, plus
  matching `without_sb` arrays.
- Julia fixture regeneration reported:
  - rectangular TM size `(4, 3)`;
  - rectangular TE component sizes `(4, 3)`, `(4, 4)`, `(4, 3)`;
  - periodic-image TM size `(4, 4)`;
  - wrote `Simulation/python/tests/fixtures/subpixel_2d_tm_v5.mat`.

Tests added/updated:
- Added `test_2d_te_cuboid_inverse_epsilon_matches_julia_fixture`.
- Added `test_2d_te_without_subpixel_smoothing_matches_julia_fixture`.
- Added `test_2d_combined_tm_te_return_matches_julia_fixture`.
- Updated unsupported-path coverage so TE is no longer expected to raise.
  The test now checks `use_2D_TM=False, use_2D_TE=False`, 3D domain rejection,
  non-`Cuboid` object rejection, and object/epsilon length mismatch.

Commands run and outcomes:
- `python -m py_compile Simulation/python/mesti/subpixel.py Simulation/python/tests/test_subpixel.py`
  - Passed.
- `wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_subpixel_2d_tm_v5_fixture.jl'`
  - Passed; regenerated `subpixel_2d_tm_v5.mat`.
  - WSL printed the usual oneAPI initialization banner and a trailing WSL
    localhost/NAT warning; fixture generation itself completed successfully.
- `python -m unittest discover -s tests -p test_subpixel.py`
  - First run passed `7 tests` but emitted `ComplexWarning` from assigning
    complex-zero Kottke helper outputs into real arrays.
  - Fixed `_tau_trans` and `_tau_inverse_trans` dtype handling to preserve
    real dtype for real inputs.
- `python -m unittest discover -s tests -p test_subpixel.py`
  - Passed cleanly: `7 tests`, `0 failures`.
- `python -m unittest discover -s tests`
  - Passed: `132 tests`, `9 skipped`.
- `codegraph sync` from `Simulation/python/mesti`
  - Synced `1 changed files`.
  - Modified `1`, `31 nodes` in `163ms`.
- Initial `codegraph status` was run in parallel with `codegraph sync` and
  reported stale pending changes. A second sequential status was run and is the
  authoritative post-sync status:
  - Project: `D:\BaiduSyncdisk\Projects\Q project\Simulation\python\mesti`
  - Files: `11`
  - Nodes: `386`
  - Edges: `1,090`
  - DB size: `1.04 MB`
  - Backend: `node:sqlite - built-in (full WAL)`
  - Status: `[OK] Index is up to date`

Known limitations from this slice:
- 2D TE support is fixture-backed for rectangular `Cuboid` domains and objects
  with scalar isotropic permittivities. Multiple-object sequencing uses the
  same tensor helper but does not yet have a dedicated Julia fixture.
- 3D tensor smoothing remains unported.
- Curved `GeometryPrimitives` shapes remain explicit unsupported Python paths
  because Python only exposes the local `Cuboid` geometry class.
- No memory or speed optimization was attempted.

Next exact action:
Continue Step 2. Choose the next smallest subpixel-completeness slice:
either 3D diagonal tensor smoothing for rectangular `Cuboid` objects with a
small Julia fixture, or explicit tested stubs/documentation for curved
`GeometryPrimitives` shapes if inspection shows 3D is too broad for the next
turn. Do not start production memory/speed optimization.

### 2026-06-05 - Opened V7 and added raw-MUMPS compatibility facade

Intent:
Start v7 from the user's changed priority: translate the whole Julia project
surface before optimizing memory/speed. Implement a concrete first completeness
slice rather than only writing a plan.

Julia sources inspected:
- `Simulation/julia/MESTI.jl-0.5.1/src/MESTI.jl`
- `Simulation/julia/MESTI.jl-0.5.1/src/mumps3_struc.jl`
- `Simulation/julia/MESTI.jl-0.5.1/src/mumps3_interface.jl`
- `Simulation/julia/MESTI.jl-0.5.1/src/mumps3_convenience_wrappers.jl`
- `Simulation/julia/MESTI.jl-0.5.1/src/mumps3_icntl_alibis.jl`
- `Simulation/julia/MESTI.jl-0.5.1/src/mumps3_printing.jl`
- Full Julia file list under `src/`, `examples/`, `MPI/`, and `mumps/` was
  scanned with `rg --files`.

Python files changed:
- Added `Simulation/python/mesti/mumps.py`.
- Updated `Simulation/python/mesti/__init__.py` to export the new facade.
- Added `Simulation/python/tests/test_mumps_compat.py`.
- Updated `Simulation/python/mesti/README.md`.
- Added this handoff file:
  `.codex/mesti_python_port_plan_v7.md`.

Implementation details:
- Added a Python `Mumps` state object that stores matrix/RHS/factorization
  state, ICNTL/CNTL/KEEP arrays, Schur state, and finalized state.
- Added Python-callable equivalents for Julia bang functions by dropping the
  bang suffix or using `_inplace`, for example:
  `set_icntl! -> set_icntl`,
  `mumps_solve! -> mumps_solve_inplace`,
  `mumps_factorize! -> mumps_factorize_inplace`,
  `mumps_det! -> mumps_det_inplace`,
  `mumps_schur_complement! -> mumps_schur_complement_inplace`,
  `mumps_select_inv! -> mumps_select_inv_inplace`,
  `initialize! -> initialize`,
  `finalize! -> finalize`.
- Working helpers use SciPy/SuperLU or dense NumPy algebra:
  `mumps_solve`, `mumps_factorize`, `mumps_det`,
  `mumps_schur_complement`, and `mumps_select_inv`.
- Control/state helpers update Python object state:
  `set_keep`, `set_icntl`, `set_cntl`, `set_job`, `set_save_dir`,
  `set_save_prefix`, `provide_matrix`, `provide_rhs`, `get_rhs`, `get_sol`,
  `get_schur_complement`, stream/print aliases, matrix/RHS mode aliases,
  predicates, and display helpers.
- Raw Julia C/MPI calls are not silently simulated:
  `invoke_mumps` and `invoke_mumps_unsafe` raise
  `UnsupportedMumpsOperation` with a message directing users to
  `mesti_matrix_solver`, `mumps_solve`, or external MUMPS bindings.
- Python low-level selected-index helpers use zero-based matrix indices,
  matching the rest of the Python package, while ICNTL/CNTL/KEEP setter indices
  remain 1-based to match the Julia MUMPS manual surface.

Tests added:
- `test_mumps_solve_matrix_rhs_matches_numpy`
- `test_mumps_object_reuses_factorization_and_tracks_rhs_solution`
- `test_mumps_solve_inplace_writes_output`
- `test_det_schur_and_selected_inverse_helpers`
- `test_schur_inplace_stores_retrievable_matrix`
- `test_controls_and_predicates_are_one_based_like_julia_docs`
- `test_finalize_blocks_later_state_mutation`
- `test_raw_mumps_invocation_is_explicit_unsupported`

Commands run and outcomes:
- `python -m unittest tests.test_mumps_compat`
  - Failed because `Simulation/python/tests` is not a Python package, so the
    module path import was invalid. No code failure.
- `python -m unittest discover -s tests -p test_mumps_compat.py`
  - Passed: `8 tests`, `0 failures`.
- `python -m unittest discover -s tests`
  - Passed: `129 tests`, `9 skipped`.
  - Local environment used SciPy fallback for solver tests.
- `codegraph sync` from `Simulation/python/mesti`
  - Synced `2 changed files`.
  - Added `1`, modified `1`, `95 nodes` in `207ms`.
- `codegraph status` from `Simulation/python/mesti`
  - Project: `D:\BaiduSyncdisk\Projects\Q project\Simulation\python\mesti`
  - Files: `11`
  - Nodes: `376`
  - Edges: `1,073`
  - DB size: `1.04 MB`
  - Backend: `node:sqlite - built-in (full WAL)`
  - Status: `[OK] Index is up to date`
- `Test-Path .codex\mesti_python_port_plan_v7.md`
  - Returned `True`.
- `Select-String` sanity check for major v7 sections
  - Confirmed `Current Status Snapshot`, `Julia-to-Python Inventory`,
    `Detailed Progress Log`, and `Next Step` are present.
- `python - <<'PY' ...`
  - Failed with a PowerShell parser error because Bash-style heredoc syntax is
    invalid in PowerShell. This was a command syntax issue only.
- `python -c "import mesti; ..."`
  - Passed and printed `mumps facade exports ok`, confirming the new facade
    exports `Mumps`, `mumps_solve`, `mumps_schur_complement`, and
    `UnsupportedMumpsOperation`.
- `Select-String` sanity check for README MUMPS docs
  - Confirmed `mumps.py`, `UnsupportedMumpsOperation`, and
    `Low-Level MUMPS` are documented.

Known limitations from this slice:
- `mesti/mumps.py` is a compatibility facade, not a production raw-MUMPS
  binding. It does not expose raw pointers, MUMPS C-library calls, distributed
  assembled matrices, elemental matrices, or MPI worker orchestration.
- `mumps_schur_complement` and `mumps_select_inv` use dense formulas and are
  intended for small translated helper scripts/tests, not optimized production
  workloads.
- Python cannot export Julia names containing `!` as normal identifiers; use
  names without `!` or `_inplace` names.

Next exact action:
Start Step 2: finish subpixel-smoothing translation completeness. First inspect
`Simulation/julia/MESTI.jl-0.5.1/src/mesti_subpixel_smoothing.jl`,
`Simulation/python/mesti/subpixel.py`,
`Simulation/python/tests/test_subpixel.py`, and
`Simulation/python/tests/fixtures/generate_subpixel_2d_tm_v5_fixture.jl`.
Then choose the smallest fixture-backed slice among:
2D TE inverse-epsilon output for rectangular cuboids,
3D diagonal tensor smoothing for rectangular cuboids, or
explicit tested stubs for unsupported curved GeometryPrimitives shapes.

## Open Decisions

- Whether V7 should make example helper functions importable from
  `mesti.examples` or keep them as script-local functions under
  `Simulation/python/examples`. Default for next chats: make reusable helpers
  importable when exact Julia function coverage is requested. Step 4 used both
  importable helpers and thin runnable scripts for raw-MUMPS demos.
- Whether `MPI/hybrid_mpi.jl` should become a runnable `mpi4py` script or a
  documented unsupported compatibility stub. Default for next chats: create an
  explicit unsupported script first unless the user requests runnable MPI.
- Whether low-level selected inverse and Schur helpers should get Julia
  fixtures. Default for next chats: add tiny fixtures only if a translated
  script calls those helpers; otherwise keep current NumPy/SciPy tests.

## Blockers

- No current blocker for V7 translation work.
- Production-size `Ws300 Ls37.5` memory failure remains a known v6/v8 blocker,
  not a v7 blocker.

## Next Step

Start Step 6. Translate standalone example helpers as reusable Python functions
or explicit script-only compatibility stubs: `asp`, `build_epsilon_disorder`,
`build_epsilon_disorder_3d`, and `plot_and_compare_distribution`. Prefer
importable helper functions when they improve exact Julia function coverage,
but keep plotting-only helpers as documented script-only or explicit
unsupported stubs if they would add heavy visual dependencies. Do not optimize
production memory/speed during this step.

## Next Prompt

```text
Continue MESTI Python port v7 from
`D:\BaiduSyncdisk\Projects\Q project`.

Read `.codex/mesti_python_port_plan_v7.md` first, especially the latest
progress log and inventory. Step 2 subpixel completeness is complete for the
current v7 scope, Step 3 raw-MUMPS facade coverage exists, and Step 4 compact
raw-MUMPS demos are ported/tested through importable helpers plus runnable
scripts. Step 5 reachable option/convenience audit is complete: direct `mesti`
now accepts Julia-style positional `Opts` overloads, `mesti2s` rejects
`opts.prefactor`, and solver unsupported options remain explicit. Start Step 6:
translate standalone example helpers as reusable Python functions or explicit
script-only compatibility stubs: `asp`, `build_epsilon_disorder`,
`build_epsilon_disorder_3d`, and `plot_and_compare_distribution`. Prefer
importable helper functions when they improve exact Julia function coverage,
but keep plotting-only helpers as documented script-only or explicit
unsupported stubs if they would add heavy visual dependencies. Do not start
production memory/speed optimization, update this v7 file after the step, run
focused tests plus the full available suite, refresh CodeGraph after source
edits, and record exact outcomes.
```
