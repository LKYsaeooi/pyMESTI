# cuDSS Through nvmath-python

This project uses NVIDIA cuDSS through `nvmath-python` when running inside the
WSL user `lky` conda environment `optical_simulation`.

The Windows Python environment is still useful for checking clean skip behavior,
but real cuDSS checks should run in WSL because that is where `nvmath-python`,
`cuda-python`, and the `nvidia-cudss-cu12` wheel are installed.

## Known Working Environment

- WSL user: `lky`
- Conda executable: `/home/lky/anaconda3/bin/conda`
- Conda environment: `optical_simulation`
- Project path in WSL:
  `/mnt/d/BaiduSyncdisk/Projects/Q project/Simulation/python`
- `nvmath-python`: `0.9.0`
- `nvidia-cudss-cu12`: `0.7.1.6`
- CUDA toolkit root: `/usr/local/cuda-12.4`
- GPU observed by the probe: `NVIDIA GeForce RTX 4060 Ti`
- Solver binding strategy reported by the project probe: `nvmath-bindings`

The probe locates:

- Python binding:
  `/home/lky/anaconda3/envs/optical_simulation/lib/python3.10/site-packages/nvmath/bindings/cudss.cpython-310-x86_64-linux-gnu.so`
- Header:
  `/home/lky/anaconda3/envs/optical_simulation/lib/python3.10/site-packages/nvidia/cu12/include/cudss.h`
- Library:
  `/home/lky/anaconda3/envs/optical_simulation/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss.so.0`
- cuDSS multithreading layer:
  `/home/lky/anaconda3/envs/optical_simulation/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0`

## PowerShell Command Pattern

Run WSL commands from the Windows project root with this shape:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project/Simulation/python" && /home/lky/anaconda3/envs/optical_simulation/bin/python -m unittest tests.test_cudss_backend'
```

Use this probe command to confirm the active environment:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project/Simulation/python" && /home/lky/anaconda3/envs/optical_simulation/bin/python -c "from mesti import cudss_backend; print(cudss_backend.probe_environment().as_dict())"'
```

Use `bash -ic` for commands that may also touch `mumpspy`, because the
interactive shell loads the user's oneAPI/MPI/MKL/MUMPS library paths. The WSL
launcher currently prints a garbled localhost/NAT warning in this desktop
environment. The command is usable when the Python command exits with status 0.

## Python Usage

The project-level availability check is:

```python
from mesti import cudss_backend

probe = cudss_backend.probe_environment()
print(probe.available)
print(probe.binding_strategy)
```

For the known WSL environment, `probe.available` should be `True` and
`probe.binding_strategy` should be `nvmath-bindings`.

The low-level NVIDIA binding imports as:

```python
from nvmath.bindings import cudss
```

`nvmath.bindings.cudss` exposes raw cuDSS API calls such as `create`,
`config_create`, `data_create`, `matrix_create_csr`, `matrix_create_dn`, and
`execute`. Keep MESTI physics logic in `mesti/solver.py` or
`mesti/cudss_backend.py`; do not bury project-specific matrix semantics inside
binding-layer calls.

## Solver Usage

Standard factorize-and-solve:

```python
from mesti import Matrices, Opts, mesti_matrix_solver

X, info = mesti_matrix_solver(Matrices(A=A, B=B), Opts(solver="cudss", verbal=False))
```

Projected APF/Schur solve:

```python
S, info = mesti_matrix_solver(
    Matrices(A=A, B=B, C=C),
    Opts(solver="cudss", method="APF", verbal=False),
)
```

The APF path builds the augmented matrix `[A B; C 0]`, asks cuDSS for the Schur
block on the appended rows/columns, and returns `C @ inv(A) @ B`. It also
supports the existing non-conjugating projection spelling `C="transpose(B)"`.

Current cuDSS Schur limitations are intentionally explicit in the backend: this
path uses one augmented system, not batched cuDSS execution, and it does not
enable cuDSS matching, COLAMD/BTF reorderings, multiblock factorization, or
MGMN/MG modes. The Schur output buffer currently requires `cuda-python`.

## Speed And Memory Controls

The cuDSS backend keeps public solver outputs as CPU-side `complex128`
`np.ndarray` objects, but selected backend work can use lower precision or
hybrid memory:

```python
X, info = mesti_matrix_solver(
    Matrices(A=A, B=B),
    Opts(solver="cudss", cudss_use_single_precision=True, verbal=False),
)
```

`cudss_use_single_precision=True` casts the cuDSS matrix and RHS to
`complex64` at the backend boundary and then casts the public result back to
`complex128`. Use looser tolerances, similar to the existing MUMPS
single-precision path.

Hybrid CPU-GPU memory mode:

```python
X, info = mesti_matrix_solver(
    Matrices(A=A, B=B),
    Opts(
        solver="cudss",
        cudss_use_hybrid_memory=True,
        cudss_hybrid_device_memory_limit="128MiB",
        cudss_register_cuda_memory=True,
        verbal=False,
    ),
)
```

`cudss_hybrid_device_memory_limit` is passed through to nvmath's
`HybridMemoryModeOptions` and may be an integer byte count or a string accepted
by nvmath, such as `"128MiB"`. Setting the memory limit or
`cudss_register_cuda_memory` also enables hybrid memory mode. These controls
are accepted only with `Opts(solver="cudss")`.

The backend also auto-detects the cuDSS multithreading layer from the installed
`nvidia-cudss-cu12`/`cu13` wheel and passes it through
`DirectSolverOptions(multithreading_lib=...)`. In the known WSL environment,
this removes nvmath's "No multithreading interface library" warning and improves
CPU-side cuDSS planning work.

## Test Guidance

Focused checks:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project/Simulation/python" && /home/lky/anaconda3/envs/optical_simulation/bin/python -m unittest tests.test_cudss_backend tests.test_solver'
```

The full WSL test suite has shown a native `Bus error` during a cumulative run
near `test_step4_spacer_both_sides_and_mixed_wavefront_match_julia_fixture`,
while that test passes in isolation. Until that native crash is isolated, use
the Windows full suite as the broad regression gate and WSL focused tests for
real cuDSS availability/dispatch checks.
