"""Reduced phase-conjugation focusing MESTI example.

This is a compact Python translation of the numerical core from Julia
``examples/2d_focusing_inside_disorder_with_phase_conjugation``. It projects a
point source inside a disorder sample onto low-side propagating channels,
phase-conjugates those coefficients, and computes regular versus
phase-conjugated focusing fields with ``mesti2s``. Plotting and GIF output are
left out of this runnable example.

Run from ``Simulation/python``:

    python examples/focusing_inside_disorder_with_phase_conjugation.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesti import (
    Opts,
    PML,
    Source_struct,
    Syst,
    mesti,
    mesti2s,
    mesti_build_channels,
    wavefront,
)


@dataclass
class PhaseConjugationFocusResult:
    """Numerical outputs from the reduced phase-conjugation focusing example."""

    projected_coefficients: np.ndarray
    projection_from_field: np.ndarray | None
    projection_from_field_difference_max: float | None
    channels_low: Any
    channels_average: Any
    regular_focus_wavefront: np.ndarray
    phase_conjugated_wavefront: np.ndarray
    v_low: np.ndarray
    field_profiles: np.ndarray
    normalized_field_profiles: np.ndarray
    normalization_factor: float
    focus_index: tuple[int, int]
    focus_intensities: np.ndarray
    phase_to_regular_intensity_ratio: float
    projection_info: Any
    projection_field_info: Any | None
    field_info: Any


def demo_disorder_permittivity(
    ny: int = 8,
    nz: int = 6,
    epsilon_bg: complex = 1.0,
    epsilon_scat: complex = 1.44,
) -> np.ndarray:
    """Build a tiny deterministic cylinder-like TM permittivity profile."""

    if ny <= 0 or nz <= 0:
        raise ValueError("ny and nz must be positive.")
    yy, zz = np.indices((ny, nz), dtype=float)
    epsilon = np.full((ny, nz), epsilon_bg, dtype=np.complex128)
    cylinders = (
        (0.30 * (ny - 1), 0.28 * (nz - 1), 1.05),
        (0.70 * (ny - 1), 0.50 * (nz - 1), 0.95),
        (0.45 * (ny - 1), 0.78 * (nz - 1), 0.85),
    )
    for y0, z0, radius in cylinders:
        mask = (yy - y0) ** 2 + (zz - z0) ** 2 <= radius**2
        epsilon[mask] = epsilon_scat
    return epsilon


def _system(
    epsilon_xx: np.ndarray,
    *,
    wavelength: float,
    dx: float,
    epsilon_low: complex,
    epsilon_high: complex,
    yBC: str | float,
    pml_npixels: int,
) -> Syst:
    return Syst(
        epsilon_xx=np.asarray(epsilon_xx, dtype=np.complex128),
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        wavelength=float(wavelength),
        dx=float(dx),
        yBC=yBC,
        zPML=[PML(int(pml_npixels))],
    )


def _projection_system(
    epsilon_xx: np.ndarray,
    *,
    wavelength: float,
    dx: float,
    epsilon_low: complex,
    epsilon_high: complex,
    yBC: str | float,
    pml_npixels: int,
) -> Syst:
    ny = epsilon_xx.shape[0]
    low_pad = np.full((ny, pml_npixels + 1), epsilon_low, dtype=np.complex128)
    high_pad = np.full((ny, pml_npixels + 1), epsilon_high, dtype=np.complex128)
    return Syst(
        epsilon_xx=np.concatenate([low_pad, epsilon_xx, high_pad], axis=1),
        wavelength=float(wavelength),
        dx=float(dx),
        yBC=yBC,
        zBC="PEC",
        PML=[PML(int(pml_npixels), direction="z")],
    )


def _default_focus_index(shape: tuple[int, int]) -> tuple[int, int]:
    return max(int(shape[0] / 2) - 1, 0), max(int(shape[1] / 2) - 1, 0)


def _point_source_projection(
    epsilon_xx: np.ndarray,
    *,
    wavelength: float,
    dx: float,
    epsilon_low: complex,
    epsilon_high: complex,
    yBC: str | float,
    pml_npixels: int,
    focus_index: tuple[int, int],
    solver: str | None,
    compute_field_projection_check: bool,
) -> tuple[np.ndarray, np.ndarray | None, float | None, Any, Any, Any | None]:
    ny = epsilon_xx.shape[0]
    focus_y, focus_z = focus_index
    if focus_y < 0 or focus_y >= epsilon_xx.shape[0] or focus_z < 0 or focus_z >= epsilon_xx.shape[1]:
        raise ValueError("focus_index must be inside epsilon_xx.")

    syst_projection = _projection_system(
        epsilon_xx,
        wavelength=wavelength,
        dx=dx,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        yBC=yBC,
        pml_npixels=pml_npixels,
    )
    source = Source_struct(
        pos=[
            np.array(
                [focus_y, int(pml_npixels) + 1 + focus_z, focus_y, int(pml_npixels) + 1 + focus_z],
                dtype=int,
            )
        ],
        data=[np.ones((1, 1), dtype=np.complex128)],
    )

    k0dx = (2 * np.pi / wavelength) * dx
    channels_low = mesti_build_channels(ny, yBC, k0dx, epsilon_low)
    projection_data = (
        np.conjugate(channels_low.f_x_m(channels_low.kydx_prop))
        * channels_low.sqrt_nu_prop.reshape(1, -1)
        * np.exp((-0.5j) * channels_low.kzdx_prop).reshape(1, -1)
    )
    projection = Source_struct(
        pos=[np.array([0, int(pml_npixels), ny - 1, int(pml_npixels)], dtype=int)],
        data=[projection_data],
    )

    projected, projection_info = mesti(
        syst_projection,
        [source],
        [projection],
        opts=Opts(solver=solver, verbal=False, use_L0_threads=False),
    )

    projection_from_field = None
    projection_field_info = None
    difference_max = None
    if compute_field_projection_check:
        point_field, projection_field_info = mesti(
            syst_projection,
            [source],
            opts=Opts(solver=solver, verbal=False, use_L0_threads=False),
        )
        projection_from_field = projection_data.T @ point_field[:, int(pml_npixels), :]
        difference_max = float(np.max(np.abs(projected - projection_from_field)))

    return (
        projected.reshape(-1),
        None if projection_from_field is None else projection_from_field.reshape(-1),
        difference_max,
        channels_low,
        projection_info,
        projection_field_info,
    )


def _regular_focus_wavefront(
    epsilon_xx: np.ndarray,
    channels_low: Any,
    *,
    wavelength: float,
    dx: float,
    yBC: str | float,
    focus_index: tuple[int, int],
) -> tuple[np.ndarray, Any]:
    k0dx = (2 * np.pi / wavelength) * dx
    channels_average = mesti_build_channels(epsilon_xx.shape[0], yBC, k0dx, np.mean(epsilon_xx))
    focus_y_julia = focus_index[0] + 1
    focus_z_julia = focus_index[1] + 1
    full_wavefront = (
        np.exp(-1j * channels_average.kydx_prop * focus_y_julia)
        * np.exp(-1j * channels_average.kzdx_prop * (focus_z_julia - 0.5))
    )
    channel_diff = int(channels_average.N_prop) - int(channels_low.N_prop)
    if channel_diff < 0 or channel_diff % 2 != 0:
        raise ValueError("Average-epsilon channels cannot be center-cropped onto low-side channels.")
    crop_each_side = channel_diff // 2
    selected = full_wavefront[crop_each_side : full_wavefront.size - crop_each_side]
    if selected.size != int(channels_low.N_prop):
        raise RuntimeError("Internal regular-focus wavefront channel crop produced the wrong size.")
    norm = np.linalg.norm(selected)
    if norm == 0:
        raise RuntimeError("Regular-focus wavefront has zero norm.")
    return selected / norm, channels_average


def run_focusing_inside_disorder_with_phase_conjugation(
    *,
    epsilon_xx: Any | None = None,
    wavelength: float = 1.0,
    dx: float = 0.25,
    epsilon_low: complex = 1.0,
    epsilon_high: complex = 1.0,
    yBC: str | float = "periodic",
    pml_npixels: int = 16,
    nz_low: int = 2,
    nz_high: int = 2,
    focus_index: tuple[int, int] | None = None,
    solver: str | None = "scipy",
    compute_field_projection_check: bool = True,
) -> PhaseConjugationFocusResult:
    """Run the reduced phase-conjugation focusing example."""

    epsilon = (
        demo_disorder_permittivity(epsilon_bg=epsilon_low, epsilon_scat=1.44)
        if epsilon_xx is None
        else np.asarray(epsilon_xx, dtype=np.complex128)
    )
    if epsilon.ndim != 2:
        raise ValueError("epsilon_xx must be a 2D TM permittivity array.")
    if pml_npixels < 0:
        raise ValueError("pml_npixels must be nonnegative.")
    if nz_low < 0 or nz_high < 0:
        raise ValueError("nz_low and nz_high must be nonnegative.")

    focus = _default_focus_index(epsilon.shape) if focus_index is None else tuple(int(v) for v in focus_index)
    (
        projected,
        projection_from_field,
        projection_difference,
        channels_low,
        projection_info,
        projection_field_info,
    ) = _point_source_projection(
        epsilon,
        wavelength=wavelength,
        dx=dx,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        yBC=yBC,
        pml_npixels=int(pml_npixels),
        focus_index=focus,
        solver=solver,
        compute_field_projection_check=compute_field_projection_check,
    )

    regular_focus, channels_average = _regular_focus_wavefront(
        epsilon,
        channels_low,
        wavelength=wavelength,
        dx=dx,
        yBC=yBC,
        focus_index=focus,
    )
    projected_norm = np.linalg.norm(projected)
    if projected_norm == 0:
        raise RuntimeError("Projected point-source coefficients have zero norm.")
    phase_conjugated = np.conjugate(projected)[channels_low.ind_prop_conj] / projected_norm

    v_low = np.zeros((int(channels_low.N_prop), 2), dtype=np.complex128)
    v_low[:, 0] = regular_focus
    v_low[:, 1] = phase_conjugated

    syst = _system(
        epsilon,
        wavelength=wavelength,
        dx=dx,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        yBC=yBC,
        pml_npixels=int(pml_npixels),
    )
    field_profiles, _, field_info = mesti2s(
        syst,
        wavefront(v_low=v_low),
        Opts(
            solver=solver,
            verbal=False,
            nz_low=int(nz_low),
            nz_high=int(nz_high),
            use_L0_threads=False,
        ),
    )
    normalization_factor = float(np.max(np.abs(field_profiles[:, :, 1])))
    if normalization_factor == 0:
        raise RuntimeError("Phase-conjugated field profile has zero maximum amplitude.")
    normalized_field_profiles = field_profiles / normalization_factor

    focus_y, focus_z = focus
    focus_z_extended = int(nz_low) + focus_z
    focus_values = field_profiles[focus_y, focus_z_extended, :]
    focus_intensities = np.abs(focus_values) ** 2
    phase_to_regular_ratio = float(focus_intensities[1] / focus_intensities[0])

    return PhaseConjugationFocusResult(
        projected_coefficients=projected,
        projection_from_field=projection_from_field,
        projection_from_field_difference_max=projection_difference,
        channels_low=channels_low,
        channels_average=channels_average,
        regular_focus_wavefront=regular_focus,
        phase_conjugated_wavefront=phase_conjugated,
        v_low=v_low,
        field_profiles=field_profiles,
        normalized_field_profiles=normalized_field_profiles,
        normalization_factor=normalization_factor,
        focus_index=focus,
        focus_intensities=focus_intensities,
        phase_to_regular_intensity_ratio=phase_to_regular_ratio,
        projection_info=projection_info,
        projection_field_info=projection_field_info,
        field_info=field_info,
    )


def _load_fixture(path: Path) -> dict[str, Any]:
    from scipy.io import loadmat

    try:
        return {
            key: value
            for key, value in loadmat(path, squeeze_me=False).items()
            if not key.startswith("__")
        }
    except NotImplementedError:
        import h5py

        data: dict[str, Any] = {}
        with h5py.File(path, "r") as handle:
            for key, value in handle.items():
                arr = np.asarray(value)
                if arr.dtype.fields and {"real", "imag"}.issubset(arr.dtype.fields):
                    arr = arr["real"] + 1j * arr["imag"]
                elif arr.dtype == np.uint16 and arr.ndim == 2 and arr.shape[1] == 1:
                    data[key] = "".join(chr(code) for code in arr.reshape(-1))
                    continue
                if arr.ndim >= 2:
                    arr = arr.T
                data[key] = arr
        return data


def _scalar(data: dict[str, Any], key: str) -> Any:
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"Fixture key {key!r} is not scalar.")
    return value.reshape(-1)[0].item()


def _fixture_string(data: dict[str, Any], key: str) -> str:
    return str(_scalar(data, key))


def run_fixture_example(path: Path | None = None) -> PhaseConjugationFocusResult:
    """Run the example against the reduced Julia-generated fixture."""

    fixture_path = (
        path
        if path is not None
        else Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "example_focusing_phase_conjugation_v5.mat"
    )
    fixture = _load_fixture(fixture_path)
    return run_focusing_inside_disorder_with_phase_conjugation(
        epsilon_xx=np.asarray(fixture["epsilon_xx"], dtype=np.complex128),
        wavelength=float(_scalar(fixture, "wavelength")),
        dx=float(_scalar(fixture, "dx")),
        epsilon_low=_scalar(fixture, "epsilon_low"),
        epsilon_high=_scalar(fixture, "epsilon_high"),
        yBC=_fixture_string(fixture, "yBC"),
        pml_npixels=int(_scalar(fixture, "pml_npixels")),
        nz_low=int(_scalar(fixture, "nz_low")),
        nz_high=int(_scalar(fixture, "nz_high")),
        focus_index=(
            int(_scalar(fixture, "focus_y_index_zero_based")),
            int(_scalar(fixture, "focus_z_index_zero_based")),
        ),
        solver="scipy",
    )


def main() -> int:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "example_focusing_phase_conjugation_v5.mat"
    )
    result = (
        run_fixture_example(fixture_path)
        if fixture_path.exists()
        else run_focusing_inside_disorder_with_phase_conjugation()
    )
    print(f"projected coefficient norm = {np.linalg.norm(result.projected_coefficients):.6f}")
    print(f"field profile shape = {result.field_profiles.shape}")
    print(f"focus intensity ratio = {result.phase_to_regular_intensity_ratio:.6f}")
    if result.projection_from_field_difference_max is not None:
        print(f"max projection-from-field difference = {result.projection_from_field_difference_max:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
