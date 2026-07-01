# MESTI Python Port

> [!WARNING]
> This project is not finished and may contain many bugs.

This project is the Python version of the MESTI.jl project: https://github.com/complexphoton/MESTI.jl/tree/main.

## Overview

This repository contains a native Python port of the MESTI Julia codebase. The
current implementation began with the 2D TM/scalar workflow needed by
[`simu_test_2D_TM.py`](simu_test_2D_TM.py): building channels, assembling a
finite-difference frequency-domain operator, solving sparse linear systems, and
returning either scattering matrices or field profiles.

The post-2D expansion now includes Julia-fixture-verified 3D vectorial FDFD
assembly, high-level direct `mesti` solves for diagonal and off-diagonal tensor
permittivity, 3D `mesti2s` channel/scattering paths, 2D TM/TE and small 3D
subpixel smoothing for rectangular `Cuboid` geometry, compact raw-MUMPS helper
demo translations, importable numerical helpers for selected examples, and an
explicit cuDSS backend for GPU-oriented sparse solves.

The public names intentionally mirror Julia names where practical. This keeps
the port easy to compare with the original implementation, while Python
conventions are used where they matter most: channel and grid indices are
zero-based, arrays are NumPy/SciPy objects, and unsupported Julia paths fail
explicitly with `NotImplementedError` or validation errors.

## Current Scope

- Core package: [`mesti/`](mesti)
- Detailed package/API documentation: [mesti/README.md](mesti/README.md)
- Examples: [`examples/`](examples)
- Regression and Julia-parity tests: [`tests/`](tests)
- cuDSS environment and backend notes: [docs/cudss_nvmath.md](docs/cudss_nvmath.md)

The project is not a full replacement for MESTI.jl yet. It is intended as a
correctness-first Python port with explicit unsupported boundaries for raw
MUMPS/MPI orchestration, broader curved-geometry subpixel smoothing, and some
advanced Julia solver controls.

## Acknowledgments

Parts of this Python port were developed with assistance from OpenAI ChatGPT/Codex.
Human review, testing, and project decisions were performed by the repository maintainer.

## Solver Backends

`Opts.solver` can select:

| Value | Behavior |
| --- | --- |
| `None` or `"auto"` | Prefer an importable Python MUMPS binding when available, otherwise use SciPy/SuperLU |
| `"scipy"` | Force SciPy/SuperLU |
| `"MUMPS"` or `"mumps"` | Use a Python MUMPS binding, preferring `mumpspy` when available |
| `"mumpspy"` | Force the `mumpspy` binding |
| `"python-mumps"` | Force the `python-mumps` binding imported as `mumps` |
| `"cudss"` | Force the explicit NVIDIA cuDSS backend through `nvmath-python` |

The cuDSS path supports factorize-and-solve and projected APF/Schur solves.
It can also use single precision at the backend boundary and nvmath hybrid
CPU-GPU memory mode. Real cuDSS runs require a working CUDA/nvmath/cuDSS
environment; see [docs/cudss_nvmath.md](docs/cudss_nvmath.md).

Minimal cuDSS usage:

```python
from mesti import Matrices, Opts, mesti_matrix_solver

X, info = mesti_matrix_solver(
    Matrices(A=A, B=B),
    Opts(solver="cudss", verbal=False),
)

S, info = mesti_matrix_solver(
    Matrices(A=A, B=B, C=C),
    Opts(solver="cudss", method="APF", verbal=False),
)
```

`simu_test_2D_TM.py` also exposes cuDSS from the command line:

```powershell
python simu_test_2D_TM.py --root "<folder-with-epsilon>" --solver cudss --method APF
```

Additional flags include `--cudss-use-single-precision`,
`--cudss-use-hybrid-memory`, `--cudss-hybrid-device-memory-limit`, and
`--cudss-no-register-cuda-memory`.

## Quick Start

Run from the repository root so `import mesti` resolves to the local package:

```powershell
python -m unittest discover -s tests
```

Minimal CPU-backed transmission example:

```python
import numpy as np

from mesti import Opts, PML, Syst, channel_type, mesti2s

syst = Syst(
    epsilon_xx=np.ones((3, 2), dtype=np.complex128),
    epsilon_low=1.0,
    epsilon_high=1.0,
    wavelength=2 * np.pi,
    dx=1.0,
    yBC="periodic",
    zPML=[PML(4)],
)

t, channels, info = mesti2s(
    syst,
    channel_type(side="low"),
    channel_type(side="high"),
    Opts(solver="scipy", verbal=False),
)

print(t.shape)
print(info.opts.solver)
```
