"""Reduced Gaussian-beam reflection-matrix MESTI example.

This is a compact runnable version of the Julia
``examples/2d_reflection_matrix_Gaussian_beams`` workflow. It uses the Python
package helper for the translated source/projection and direct ``mesti`` solve
logic, leaving plotting out of the example.

Run from ``Simulation/python``:

    python examples/reflection_matrix_gaussian_beams.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesti import reflection_matrix_gaussian_beams


def run_reflection_matrix_gaussian_beams():
    """Run a small Gaussian-beam reflection example."""

    wavelength = 1.0
    dx = 0.25
    epsilon_xx = np.ones((15, 11), dtype=np.complex128)
    epsilon_xx[5:10, 5:7] = 1.2**2
    return reflection_matrix_gaussian_beams(
        epsilon_xx=epsilon_xx,
        wavelength=wavelength,
        dx=dx,
        pml_npixels=2,
        y_focus=[1.4, 2.0, 2.6],
        z_focus=1.5,
        source_plane_index=2,
        epsilon_bg=1.0,
        numerical_aperture=0.5,
        solver="scipy",
    )


def main() -> int:
    result = run_reflection_matrix_gaussian_beams()
    singular_values = np.linalg.svd(result.reflection, compute_uv=False)
    print(f"reflection shape = {result.reflection.shape}")
    print(f"field profile shape = {result.field_profiles.shape}")
    print(f"reflection singular values = {singular_values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
