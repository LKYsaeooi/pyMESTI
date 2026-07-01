"""Compact Python analogue of Julia ``mumps/schur_complement.jl``.

Run from ``Simulation/python``:

    python examples/mumps_schur_complement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesti import mumps_schur_complement_demo


def run_mumps_schur_complement(*, m: int = 6, n: int = 3, dtype=np.complex128):
    """Return the deterministic Schur-complement demo result."""

    return mumps_schur_complement_demo(m=m, n=n, dtype=dtype)


def main() -> None:
    for dtype in (np.complex128, np.complex64):
        result = run_mumps_schur_complement(dtype=dtype)
        print(
            f"{result.dtype.name}: relative_error={result.relative_error:.3e}, "
            f"tolerance={result.relative_tolerance:.3e}, "
            f"indices={result.schur_indices.tolist()}"
        )


if __name__ == "__main__":
    main()
