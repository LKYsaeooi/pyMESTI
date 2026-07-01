# MESTI Regression Fixtures

`mesti2s_2d_tm_python_fixture.json` is a deterministic Python-generated
regression fixture for the 2D TM MESTI port. It is not a Julia parity fixture;
it remains as a historical Python baseline from before WSL Julia fixture
generation was wired into this project.

The fixture locks down a small periodic 2D TM system with three propagating
channels on both the low and high sides. It stores:

- the scattering system parameters,
- the low-side wavefront input,
- the low-to-high transmission matrix,
- the corresponding field profile, and
- selected channel metadata.

Regenerate it from `Simulation/python` with:

```powershell
conda run -n simu_scattering_light python tests\fixtures\generate_mesti2s_python_fixture.py
```

Do not overwrite this Python baseline with Julia data; use the separate
Julia-generated fixtures below for parity coverage.

## V5 Fixture Suite Status

The v5 port phase is complete. Its Julia-generated fixtures now cover the
accepted closeout scope: diagonal and off-diagonal 3D FDFD/direct/`mesti2s`
paths, 3D homogeneous field extension, one-sided off-diagonal low reflection,
2D TM symmetrized-K and APF-default solver behavior, explicit low-level SciPy
FG parity, rectangular-Cuboid 2D TM/TE subpixel smoothing, and the reduced
packaged example translations. Later v7 fixtures added rectangular-Cuboid 3D
subpixel smoothing for both face-planar and edge/corner cuts. Remaining
fixture-backed candidates for a maintenance phase are `python-mumps` APF/raw
MUMPS controls, out-of-core MUMPS behavior, curved-shape smoothing beyond the
explicit Python `Ball` unsupported stub, and any new production-scale solver
route. V6 added the first explicit `mumpspy` single-precision fixture and
diagnostics.

## Julia Parity Fixtures

`mesti2s_2d_tm_julia_low_to_high.mat` and
`mesti2s_2d_tm_julia_wavefront_v_low.mat` are Julia-generated parity fixtures
for the same deterministic 2D TM system. They lock down:

- the low-to-high transmission matrix from
  `mesti2s(syst, channel_type("low"), channel_type("high"))`, and
- the field profile from `mesti2s(syst, wavefront(v_low=...))`.

Regenerate them from the project root with the WSL Julia environment:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_julia_fixtures.jl'
```

Use `bash -ic` so the WSL shell loads the Julia, Intel oneAPI, MPI, MKL, and
MUMPS environment configured in the user's shell startup files.

The MAT files store Julia's one-based channel indices as well as zero-based
copies used by the Python tests. The parity tests use `rtol = 5e-5` and
`atol = 2e-6` because Julia/MUMPS and Python/SciPy use different sparse direct
solver stacks.

`mesti2s_2d_tm_ws30_center384_double_mumps.mat` is a cropped-real Julia parity
fixture from the documented `Ws30 Ls7.5` input. It stores the centered
`384 x 120` permittivity crop, the low-to-high transmission matrix, a fixed
open-channel input wavefront, and the corresponding field profile. This fixture
uses `opts.use_single_precision_MUMPS = false` in Julia so it remains a tight
double-MUMPS reference for Python's default MUMPS behavior. Explicit Python
`mumpspy` single precision is tested separately because the same cropped case
differs from this double reference at about the `2e-4` relative level.

Regenerate the cropped-real fixture from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_ws30_cropped_julia_fixture.jl'
```

The cropped-real Python test skips automatically when no Python MUMPS binding
is importable, so ordinary Windows SciPy-only test runs remain lightweight.

For v6 single-precision `mumpspy` diagnostics against this same cropped-real
fixture, run from `Simulation/python` in the WSL base Python environment:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project/Simulation/python" && python tests/fixtures/run_ws30_single_precision_v6_diagnostic.py transmission'
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project/Simulation/python" && python tests/fixtures/run_ws30_single_precision_v6_diagnostic.py field'
```

The diagnostic prints JSON metrics for solver settings, elapsed time, peak RSS
when available, and numerical drift versus the Julia double-MUMPS reference. It
does not write production output files. The v6 recorded WSL metrics were:
transmission relative drift `1.83e-4`, singular-value relative drift
`1.47e-5`, field-profile relative drift `1.91e-4`, and sub-300 MB peak RSS on
the centered crop.

## Step 4 Julia Parity Fixtures

`generate_mesti2s_step4_julia_fixtures.jl` writes five tiny Julia-generated
fixtures that expand coverage for less common 2D TM paths:

- `mesti2s_2d_tm_step4_bloch_continuous.mat` covers `syst.ky_B`, Bloch
  channel phases, `opts.use_continuous_dispersion`, and nonzero `opts.m0`.
- `mesti2s_2d_tm_step4_nonperiodic.mat` covers a non-periodic transverse
  `PMC` boundary.
- `mesti2s_2d_tm_step4_spacer_wavefront.mat` covers nonzero
  `PML.npixels_spacer`, `channel_type(side="both")` with both sides
  propagating, high-side incident content, and mixed low/high wavefront inputs.
- `mesti2s_2d_tm_step4_interface_rt.mat` follows Julia's
  `interface_t_r_test.jl` one-interface reflection/transmission setup.
- `mesti_step4_direct_2d_tm.mat` covers direct `mesti` field and projection
  paths.

Regenerate the Step 4 fixture bundle from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_step4_julia_fixtures.jl'
```

These fixtures are generated with Julia `opts.use_single_precision_MUMPS =
false`. The systems are small enough for Python/SciPy parity tests, so they do
not require Python MUMPS bindings at runtime.

## V5 2D TM Symmetrized-K Fixture

`mesti2s_2d_tm_symmetrized_k_v5.mat` is a tiny Julia-generated parity fixture
for high-level `mesti2s` with `opts.symmetrize_K = true`. It stores asymmetric
low/high channel-index selections, Julia's conjugate-channel permutation,
expanded padded solve-channel lists, restored input/output positions, and the
MUMPS/APF symmetrized result alongside an unsymmetrized direct reference.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_symmetrized_k_v5_fixture.jl'
```

The Python parity test runs through SciPy/SuperLU with `C = "transpose(B)"`,
so it verifies wrapper channel padding and restoration without requiring a
Python MUMPS binding at runtime. Explicit `Opts.method = "APF"` still requires
the supported `mumpspy` backend in Python.

`mesti2s_2d_tm_apf_default_v5.mat` is a tiny Julia-generated parity fixture
for the high-level 2D TM projected-solve default that matters for production
memory behavior. Julia MESTI uses MUMPS/APF by default for scattering-matrix
`mesti2s` calls when MUMPS is available; this fixture stores that APF
transmission matrix, a factorize-and-solve reference on the same deterministic
system, singular values, and channel metadata.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_apf_default_v5_fixture.jl'
```

The Windows-safe Python test compares SciPy factorize-and-solve to the Julia
APF reference numerically. A WSL-only test additionally verifies that Python
high-level `mesti2s` defaults to `mumpspy` APF for projected 2D TM solves when
that backend is importable.

## V5 Low-Level Solver FG Fixture

`solver_fg_v5.mat` is a tiny Julia-generated parity fixture for low-level
`mesti_matrix_solver!` with `opts.solver = "JULIA"` and `opts.method = "FG"`.
It stores an ordinary sparse projected solve, a `C = "transpose(B)"`
non-conjugating transpose case, factorize-and-solve references for both, and
singular values.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_solver_fg_v5_fixture.jl'
```

The Python parity tests run through explicit `Opts(solver="scipy",
method="FG")`. MUMPS-backed FG and `python-mumps` APF remain unsupported
because the verified Python bindings do not expose the required raw grouping or
Schur-complement surfaces.

## V6 Low-Level MUMPS Single-Precision Fixture

`solver_mumps_single_precision_v6.mat` is a tiny Julia-generated parity
fixture for low-level `mesti_matrix_solver!` with `opts.solver = "MUMPS"` and
explicit `opts.use_single_precision_MUMPS = true`. It stores ordinary
factorize-and-solve, projected factorize-and-solve, and APF projected results,
plus double-MUMPS references on the same matrix so the tests can record
single-vs-double drift.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_solver_mumps_single_precision_v6_fixture.jl'
```

The Python parity tests require `mumpspy` and skip on Windows environments
where no `mumpspy` binding is installed. They compare Python's explicit
`Opts(use_single_precision_MUMPS = true)` route to the Julia single-MUMPS
fixture with `rtol = 5e-5` and `atol = 5e-6`; the stored tiny-case
single-vs-double drifts are expected to remain below `1e-4`. This fixture does
not claim production-size memory parity.

## Step 7 3D FDFD Fixture

`fdfd_3d_diagonal_pec.mat` is a Julia-generated parity fixture for the first
post-2D vectorial assembly slice. It stores a tiny PEC-bounded 3D Yee grid with
diagonal tensor permittivity arrays `epsilon_xx`, `epsilon_yy`, and
`epsilon_zz`, plus Julia's dense `mesti_build_fdfd_matrix` operator. The
fixture checks Python's component stacking order `[Ex[:]; Ey[:]; Ez[:]]`,
column-major flattening, curl-block signs, and Yee-grid staggering for the
diagonal tensor baseline.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_fdfd_3d_diagonal_julia_fixture.jl'
```

## V5 3D FDFD Boundary/PML Fixtures

`fdfd_3d_diagonal_v5_boundaries.mat` is a Julia-generated parity fixture bundle
for already implemented diagonal 3D FDFD branches that were not covered by the
original PEC-only fixture. It stores four tiny dense matrix references:

- `pml`: periodic 3D grid with one UPML pixel on both sides of x, y, and z.
- `bloch`: numeric Bloch phases in x and y with periodic z.
- `mixed_bc`: non-periodic `PMC`, `PECPMC`, and `PMCPEC` boundary coverage.
- `sc_pml`: periodic 3D grid with one SC-PML pixel on both sides of x, y, and z
  through `use_UPML=false`.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_fdfd_3d_diagonal_v5_fixtures.jl'
```

The Python tests compare matrix shape, nonzero count, `is_symmetric_A`, returned
PML pixel counts, and dense matrix values with `rtol = 2e-12` and `atol =
2e-12`.

## V5 3D Off-Diagonal FDFD Fixtures

`fdfd_3d_offdiagonal_v5.mat` is a Julia-generated parity fixture bundle for
the six off-diagonal 3D tensor couplings in `mesti_build_fdfd_matrix`. It
stores four tiny dense matrix references:

- `hermitian`: no-PML periodic tensor with conjugate-paired off-diagonal terms.
- `lossy`: asymmetric/lossy tensor terms that Julia accepts with warnings.
- `pml`: periodic tensor system with one UPML pixel on both sides of x, y, and
  z.
- `mixed_bc`: non-periodic `PMC`, `PECPMC`, and `PMCPEC` boundary coverage with
  Yee-staggered component shapes.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_fdfd_3d_offdiagonal_v5_fixtures.jl'
```

The Python tests compare matrix shape, nonzero count, `is_symmetric_A`, returned
PML pixel counts, and dense matrix values with `rtol = 2e-12` and `atol =
2e-12`.

## V3 3D Direct MESTI Fixture

`mesti_3d_direct_diagonal_pec.mat` is a Julia-generated parity fixture for the
first high-level 3D direct `mesti` slice. It uses the same tiny no-PML,
PEC-bounded diagonal tensor grid style as the Step 7 FDFD fixture, then stores:

- dense RHS field profiles `Ex`, `Ey`, and `Ez`,
- a dense projected solve with a nonzero `D` subtraction,
- equivalent component `Source_struct` RHS field profiles, and
- an equivalent component `Source_struct` projection.

The fixture generator uses Julia's full-cuboid `Source_struct.pos` form for
component source/projection references. Python tests convert that behavior to
the port's zero-based inclusive 3D `Source_struct.pos` convention
`[x1, y1, z1, x2, y2, z2]` and zero-based component-local `ind` convention.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti_3d_direct_julia_fixture.jl'
```

## V5 3D Direct `mesti` Boundary/PML Fixtures

`mesti_3d_direct_v5_boundaries.mat` is a Julia-generated direct-solve fixture
bundle for already implemented diagonal 3D direct `mesti` branches that were
not covered by the original PEC-only fixture. It stores dense RHS inputs and
`Ex`, `Ey`, and `Ez` field-profile references for:

- `pml`: periodic 3D grid with one PML pixel on both sides of x, y, and z.
- `bloch`: numeric Bloch phases in x, y, and z.
- `mixed_bc`: non-periodic `PMC`, `PECPMC`, and `PMCPEC` boundary coverage.

The generator uses Julia's low-level `mesti_build_fdfd_matrix` plus direct
`A \ B` solves instead of Julia high-level dense-B direct `mesti`, because
Julia MESTI 0.5.1 has a known high-level dense-B reference issue for this path.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti_3d_direct_v5_fixtures.jl'
```

## V5 3D Direct Off-Diagonal `mesti` Fixtures

`mesti_3d_direct_offdiagonal_v5.mat` is a Julia-generated direct-solve fixture
bundle for off-diagonal 3D tensor `mesti` coverage. It stores dense RHS inputs,
dense projection matrices, nonzero `D` matrices, `Ex`, `Ey`, and `Ez`
field-profile references, and `C*(A\B)-D` projected-solve references for:

- `hermitian`: no-PML periodic tensor with conjugate-paired off-diagonal terms.
- `lossy`: asymmetric/lossy tensor terms that Julia accepts with warnings.
- `pml`: periodic tensor system with one UPML pixel on both sides of x, y, and
  z.

The generator uses Julia's low-level `mesti_build_fdfd_matrix` plus direct
`A \ B` solves instead of Julia high-level dense-B direct `mesti`, because
Julia MESTI 0.5.1 has a known high-level dense-B reference issue for this path.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti_3d_direct_offdiagonal_v5_fixture.jl'
```

## V5 Direct `mesti` Option-Surface Fixtures

`mesti_direct_options_v5.mat` is a Julia-generated direct-solve fixture bundle
for Step 7 option-surface parity. It stores dense RHS inputs and direct
references for:

- 2D SC-PML with default `PEC` boundary parsing, `Opts.prefactor`,
  `Opts.exclude_PML_in_field_profiles`, and `C = "transpose(B)"` with a
  nonzero `D` matrix.
- 2D `ky_B` and `kz_B` convenience parsing, stored as equivalent dimensionless
  low-level Bloch phases in the Julia reference.
- 3D SC-PML with `direction="all"` PML parsing, `Opts.prefactor`, and PML
  exclusion on returned `Ex`, `Ey`, and `Ez` field profiles.

The generator uses Julia's low-level `mesti_build_fdfd_matrix` plus direct
`A \ B` solves instead of Julia high-level dense-B direct `mesti`, because
Julia MESTI 0.5.1 has a known high-level dense-B reference issue for this path.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti_direct_options_v5_fixture.jl'
```

## V5 2D TM/TE Subpixel Smoothing Fixture

`subpixel_2d_tm_v5.mat` is a Julia-generated parity fixture for the first
subpixel-smoothing slice. It stores 2D TM `epsilon_xx` and 2D TE
inverse-epsilon outputs from `mesti_subpixel_smoothing` for axis-aligned
`GeometryPrimitives.Cuboid` rectangles:

- a smoothed rectangle with `zBC = "PEC"` boundary cropping,
- the same rectangle with `without_sb = true`,
- TE `inv_epsilon_yy`, `inv_epsilon_zz`, and `inv_epsilon_yz` component
  outputs for both smoothed and `without_sb = true` cases, and
- a periodic-boundary image case for an object crossing the low-y domain edge.

The Python tests compare the returned arrays with `rtol = 1e-12` and `atol =
1e-12`. 3D tensor smoothing is covered separately by the v7 Cuboid fixture.
Curved-shape smoothing remains an explicit unsupported path; Python exposes
`Ball` only as a compatibility stub.

Regenerate it from the project root with:

```powershell
    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_subpixel_2d_tm_v5_fixture.jl'
```

## V7 3D Cuboid Subpixel Smoothing Fixture

`subpixel_3d_cuboid_v7.mat` is a Julia-generated parity fixture for rectangular
3D subpixel smoothing. It stores all nine tensor components returned by
`mesti_subpixel_smoothing` for a small face-planar rectangular
`GeometryPrimitives.Cuboid` slab and a finite Cuboid that exercises edge/corner
volume-fraction cuts:

- smoothed 3D tensor components `epsilon_xx`, `epsilon_xy`, `epsilon_xz`,
  `epsilon_yx`, `epsilon_yy`, `epsilon_yz`, `epsilon_zx`, `epsilon_zy`, and
  `epsilon_zz`,
- matching components with `without_sb = true`, and
- periodic boundaries in all three directions to keep both component shapes
  equal at `(2, 2, 2)`.

The edge/corner fixture pins the Kottke local-plane `surfpt_nearby`/`volfrac`
behavior for rectangular Cuboids. Curved-shape smoothing remains outside the
current Python scope; Python exposes `Ball` only as an explicit unsupported
compatibility stub.

Regenerate it from the project root with:

```powershell
    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_subpixel_3d_cuboid_v7_fixture.jl'
```

## V5 Packaged Gaussian-Beam Example Fixture

`example_reflection_gaussian_beams_v5.mat` is a reduced Julia-generated
fixture for the packaged example
`examples/2d_reflection_matrix_Gaussian_beams`. It keeps the example's
Gaussian line-source construction, `C = "transpose(B)"` reciprocity shortcut,
homogeneous-reference subtraction, all-side PML, and PML-excluded field-profile
return, but uses a small deterministic grid suitable for Python/SciPy tests.

The Julia example uses `GeometryPrimitives.Ball` to generate the circular
scatterer. Python exposes `Ball` only as an explicit unsupported compatibility
stub and does not yet port curved-shape subpixel smoothing, so this fixture
stores Julia's Ball-smoothed `epsilon_xx`; the Python test validates the
translated source/projection and solve path against that stored permittivity.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_example_reflection_gaussian_beams_v5_fixture.jl'
```

The Python tests compare `B_low`, `C_low`, the homogeneous reference matrix,
the reflection matrix, reflection singular values, and the returned field
profile with `rtol = 5e-8` and `atol = 5e-9`.

## V5 Packaged Open-Channel Disorder Example Fixture

`example_open_channel_through_disorder_v5.mat` is a reduced Julia-generated
fixture for the packaged example `examples/2d_open_channel_through_disorder`.
It keeps the low-to-high transmission matrix, SVD open-channel selection,
normal-plane-wave versus open-channel field profiles, and the direct
`mesti`-source field comparison, but uses a small deterministic random-cylinder
system suitable for Python/SciPy tests.

The Julia example uses `GeometryPrimitives.Ball` through
`build_epsilon_disorder.jl` to generate the scatterers. Python exposes `Ball`
only as an explicit unsupported compatibility stub and does not yet port
curved-shape subpixel smoothing, so this fixture stores Julia's smoothed
`epsilon_xx`; the standalone Python example validates the translated `mesti2s`,
SVD, wavefront, and direct-source logic against that stored permittivity.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_example_open_channel_through_disorder_v5_fixture.jl'
```

The Python tests compare the transmission matrix, singular values,
transmission metrics, phase-adjusted open-channel vector/field, and direct
field comparison with `rtol = 5e-8` and `atol = 5e-9`.

## V5 Packaged Phase-Conjugation Focusing Example Fixture

`example_focusing_phase_conjugation_v5.mat` is a reduced Julia-generated
fixture for the packaged example
`examples/2d_focusing_inside_disorder_with_phase_conjugation`. It keeps the
point-source projection onto low-side propagating channels, the direct
field-profile projection check, regular-focus and phase-conjugated incident
wavefront construction, and the extended `mesti2s` field-profile comparison,
but uses a small deterministic random-cylinder system suitable for
Python/SciPy tests.

The Julia example uses `GeometryPrimitives.Ball` through
`build_epsilon_disorder.jl` to generate the scatterers. Python exposes `Ball`
only as an explicit unsupported compatibility stub and does not yet port
curved-shape subpixel smoothing, so this fixture stores Julia's smoothed
`epsilon_xx`; the standalone Python example validates the translated direct
`mesti` projection and `mesti2s` wavefront-field logic against that stored
permittivity.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_example_focusing_phase_conjugation_v5_fixture.jl'
```

The Python tests compare the projected point-source coefficients, projection
from the direct field, regular and phase-conjugated wavefronts, raw and
normalized field profiles, focus intensities, and focus-intensity ratio with
`rtol = 5e-8` and `atol = 5e-9`.

## V5 Packaged Metalens ASP Example Fixture

`example_metalens_asp_v5.mat` is a reduced Julia-generated fixture for the
packaged example
`examples/2d_metalens_focusing_via_angular_spectrum_propagation`. It keeps the
truncated incident plane-wave construction, direct `mesti` solve that samples
the transmitted field immediately after the metalens, and angular spectrum
propagation to a focal plane, but replaces the production-size metalens design
file with a tiny deterministic `8 x 4` lens region suitable for Python/SciPy
tests.

Plotting, animation, internal-metalens field movies, and the original
production `permittivity_of_metalens.mat` behavior are not part of this
fixture. The standalone Python example validates source/projection indexing,
continuous-dispersion channel inputs, ASP sampling/padding, focal-plane fields,
and intensity metrics against the stored Julia outputs.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_example_metalens_asp_v5_fixture.jl'
```

The Python tests compare the truncated source profiles, sampled transmitted
field, ASP focal-plane field, focal-plane intensity, target focal intensities,
and peak metrics with `rtol = 5e-8` and `atol = 5e-9`.

## V5 Packaged 3D Open-Channel Disorder Example Fixture

`example_3d_open_channel_through_disorder_v5.mat` is a reduced Julia-generated
fixture for the packaged example `examples/3d_open_channel_through_disorder`.
It keeps the 3D `mesti2s` low-to-high both-polarization transmission matrix,
SVD closed/open-channel selection, and `Ex`/`Ey`/`Ez` field-profile workflow
for closed-channel, open-channel, and normal p-polarized plane-wave inputs, but
replaces the production random Ball-smoothed 3D tensor disorder with a tiny
deterministic lossy diagonal tensor system suitable for Python/SciPy tests.

Plotting, histogram/DMPK comparison, random non-overlapping scatterer
generation, curved 3D subpixel smoothing, out-of-core MUMPS, and the original
hour-scale production geometry are not part of this fixture. The standalone
Python example validates 3D s/p channel ordering, SVD wavefront splitting,
transmission metrics, raw field profiles, and combined closed/open fields
against the stored Julia outputs. SVD-derived vectors and fields are compared
up to a global complex phase.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_example_3d_open_channel_through_disorder_v5_fixture.jl'
```

The Python tests compare transmission, singular values, transmission
eigenvalues, plane-wave/closed/open transmission metrics, the normal
p-polarized field, combined closed/open fields, and normalization metadata with
`rtol = 5e-8` and `atol = 5e-9`.

## V4 3D `mesti2s` Fixture

`mesti2s_3d_diagonal_periodic.mat` is a Julia-generated parity fixture for the
first diagonal 3D `mesti2s` scope. It stores a tiny periodic transverse system
with diagonal `epsilon_xx`, `epsilon_yy`, and `epsilon_zz`, no z-PML pixels,
and five propagating transverse channels per side. The fixture covers:

- 3D channel metadata with x-fastest propagating channel ordering,
- two-sided s/p scattering with `channel_type(side="both", polarization="both")`,
- zero-based Python `channel_index` subselect parity against Julia one-based
  indices,
- mixed low/high s/p `wavefront` input with default scattering-region
  `Ex`, `Ey`, and `Ez` field profiles, and
- one-sided low reflection. Julia MESTI 0.5.1 has a high-level one-sided 3D
  `mesti2s` bug (`N_prop_high` is referenced before definition), so the
  generator records this reference through the same Julia channel and direct
  `mesti` formulas explicitly.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_3d_diagonal_julia_fixtures.jl'
```

The Python parity tests use `rtol = 5e-8` and `atol = 5e-9`.

## V5 3D `mesti2s` Boundary/PML Fixtures

`mesti2s_3d_diagonal_v5_boundaries.mat` is a Julia-generated two-sided
diagonal 3D `mesti2s` fixture bundle for branches that were not covered by the
v4 periodic/no-PML fixture. It stores:

- `pml`: periodic transverse boundaries with one z-PML pixel on each side.
- `bloch`: numeric Bloch phases in x and y, generated through Julia `kx_B` and
  `ky_B` and stored as equivalent dimensionless Python boundary phases.
- `mixed_bc`: non-periodic transverse `PMC` and `PECPMC` boundaries.

Each case has five propagating channels per side and a `20 x 20` two-sided
s/p scattering matrix from `channel_type(side="both", polarization="both")`.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_3d_diagonal_v5_fixtures.jl'
```

The Python parity tests use `rtol = 5e-8` and `atol = 5e-9`.

## V5 3D `mesti2s` Homogeneous Field-Extension Fixture

`mesti2s_3d_nz_extension.mat` is a corrected-formula Julia fixture for
diagonal 3D `mesti2s` field profiles with nonzero `opts.nz_low` and
`opts.nz_high`. Julia MESTI 0.5.1 has broken high-level references in this
extension branch, so the generator uses high-level Julia `mesti2s` only for the
verified surface-field profiles with `nz_low = nz_high = 1`, then applies the
corrected formulas from `src/mesti2s.jl` locally. The one-sided case uses a
manual low-level Julia `mesti` source construction because upstream one-sided
3D `mesti2s` references `N_prop_high` before definition.

The fixture covers:

- two-sided `channel_type(side="both", polarization="both")` field extension,
- mixed low/high s/p `wavefront` input with low and high homogeneous pixels,
- one-sided low s/p `wavefront` input with high-side zero padding, and
- `Ex`, `Ey`, and `Ez` shapes including the Yee-staggered high-side `Ez`
  surface.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_3d_nz_extension_julia_fixture.jl'
```

The Python parity tests use `rtol = 5e-8` and `atol = 5e-9`.

## V5 3D Off-Diagonal `mesti2s` Fixture

`mesti2s_3d_offdiagonal_v5.mat` is a Julia-generated 3D `mesti2s` fixture for
Hermitian tensor permittivity with all six off-diagonal components present. It
stores:

- diagonal and off-diagonal tensor arrays,
- channel metadata for `channel_type(side="both", polarization="both")`,
- the `4 x 4` two-sided s/p scattering matrix,
- the manual `2 x 2` one-sided low-reflection s/p scattering matrix,
- scattering singular values, and
- finite-PML unitarity metrics.

The off-diagonal tensor components are padded with zero homogeneous slabs in
the low/high regions before the Python wrapper delegates to direct `mesti`,
matching Julia's high-level two-sided path. The one-sided reference uses the
manual direct-source workaround for Julia MESTI 0.5.1's `N_prop_high` bug and
follows Julia's one-sided source branch, which fills the low off-diagonal
homogeneous slabs with `epsilon_low`.

Regenerate it from the project root with:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_3d_offdiagonal_v5_fixture.jl'
```

The Python parity tests use `rtol = 5e-8` and `atol = 5e-9`. With 16 z-PML
pixels, the recorded Julia unitarity residual is about `8.34e-5`.
