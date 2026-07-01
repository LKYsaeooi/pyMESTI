"""Compact Python analogue of Julia ``mumps/basic_solve.jl``.

Run from ``Simulation/python``:

    python examples/basic_mumps_solve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesti import basic_mumps_solve_demo


def run_basic_mumps_solve(
    *,
    n: int = 12,
    nrhs: int = 3,
    dtype=np.complex128,
    sparse_rhs: bool = False,
):
    """Return the deterministic sparse multi-RHS solve demo result."""

    return basic_mumps_solve_demo(n=n, nrhs=nrhs, dtype=dtype, sparse_rhs=sparse_rhs)


def main() -> None:
    for dtype, sparse_rhs in ((np.complex128, False), (np.complex64, True)):
        result = run_basic_mumps_solve(dtype=dtype, sparse_rhs=sparse_rhs)
        print(
            f"{result.dtype.name} sparse_rhs={result.sparse_rhs}: "
            f"residual={result.residual_norm:.3e}, "
            f"tolerance={result.residual_tolerance:.3e}"
        )


if __name__ == "__main__":
    main()
