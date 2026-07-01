"""Reduced open-channel-through-disorder MESTI example.

This is a compact Python translation of the numerical core from Julia
``examples/2d_open_channel_through_disorder``. It computes a low-to-high
transmission matrix with ``mesti2s``, extracts the most-open incident
wavefront by SVD, computes field profiles for a normal plane wave and the open
channel, and optionally reproduces the same wavefront field through direct
``mesti`` source construction.

Run from ``Simulation/python``:

    python examples/open_channel_through_disorder.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesti import Opts, PML, Source_struct, Syst, channel_type, mesti, mesti2s, wavefront


@dataclass
class OpenChannelResult:
    """Numerical outputs from the reduced open-channel example."""

    transmission: np.ndarray
    channels: Any
    singular_values: np.ndarray
    transmission_eigenvalues: np.ndarray
    open_channel: np.ndarray
    normal_index: int
    v_low: np.ndarray
    average_transmission: float
    plane_wave_transmission: float
    open_channel_transmission: float
    field_profiles: np.ndarray
    direct_field_profiles: np.ndarray | None
    direct_field_difference_max: float | None
    transmission_info: Any
    field_info: Any
    direct_info: Any | None


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
        (0.30 * (ny - 1), 0.30 * (nz - 1), 1.05),
        (0.68 * (ny - 1), 0.48 * (nz - 1), 0.95),
        (0.42 * (ny - 1), 0.74 * (nz - 1), 0.85),
    )
    for y0, z0, radius in cylinders:
        mask = (yy - y0) ** 2 + (zz - z0) ** 2 <= radius**2
        epsilon[mask] = epsilon_scat
    return epsilon


def normal_incidence_index(n_prop: int) -> int:
    """Return the zero-based normal-incidence channel index used by Julia."""

    if n_prop <= 0:
        raise ValueError("At least one propagating channel is required.")
    return int(round((n_prop + 1) / 2)) - 1


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


def _direct_mesti_wavefront_field(
    epsilon_xx: np.ndarray,
    channels: Any,
    v_low: np.ndarray,
    *,
    wavelength: float,
    dx: float,
    epsilon_low: complex,
    epsilon_high: complex,
    yBC: str | float,
    pml_npixels: int,
    solver: str | None,
) -> tuple[np.ndarray, Any]:
    ny = epsilon_xx.shape[0]
    low_pad = np.full((ny, pml_npixels + 1), epsilon_low, dtype=np.complex128)
    high_pad = np.full((ny, pml_npixels + 1), epsilon_high, dtype=np.complex128)
    direct_syst = Syst(
        epsilon_xx=np.concatenate([low_pad, epsilon_xx, high_pad], axis=1),
        wavelength=float(wavelength),
        dx=float(dx),
        yBC=yBC,
        zBC="PEC",
        PML=[PML(int(pml_npixels), direction="z")],
    )

    f_prop_low = channels.f_x_m(channels.low.kydx_prop)
    source_weights = (
        channels.low.sqrt_nu_prop
        * np.exp((-0.5j) * channels.low.kzdx_prop)
    ).reshape(-1, 1)
    source_data = f_prop_low @ (source_weights * v_low)
    source = Source_struct(
        pos=[np.array([0, int(pml_npixels), ny - 1, int(pml_npixels)], dtype=int)],
        data=[source_data],
    )
    return mesti(
        direct_syst,
        source,
        opts=Opts(solver=solver, verbal=False, prefactor=-2j, use_L0_threads=False),
    )


def _comparison_slices(
    field_profiles: np.ndarray,
    direct_field_profiles: np.ndarray,
    *,
    nz_low: int,
    nz_high: int,
    pml_npixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    mesti2s_core = field_profiles[:, nz_low : field_profiles.shape[1] - nz_high - 1, :]
    direct_core = direct_field_profiles[
        :,
        pml_npixels + 1 : direct_field_profiles.shape[1] - pml_npixels - 2,
        :,
    ]
    if mesti2s_core.shape != direct_core.shape:
        raise RuntimeError(
            "Direct mesti and mesti2s comparison slices have different shapes: "
            f"{mesti2s_core.shape} versus {direct_core.shape}."
        )
    return mesti2s_core, direct_core


def run_open_channel_through_disorder(
    *,
    epsilon_xx: Any | None = None,
    wavelength: float = 1.0,
    dx: float = 0.25,
    epsilon_low: complex = 1.0,
    epsilon_high: complex = 1.0,
    yBC: str | float = "periodic",
    pml_npixels: int = 2,
    nz_low: int = 2,
    nz_high: int = 2,
    solver: str | None = "scipy",
    compute_direct_mesti_source: bool = True,
) -> OpenChannelResult:
    """Run the reduced open-channel example and return numerical results."""

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

    syst = _system(
        epsilon,
        wavelength=wavelength,
        dx=dx,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        yBC=yBC,
        pml_npixels=pml_npixels,
    )
    transmission, channels, transmission_info = mesti2s(
        syst,
        channel_type(side="low"),
        channel_type(side="high"),
        Opts(solver=solver, verbal=False, use_L0_threads=False),
    )
    _, singular_values, vh = np.linalg.svd(transmission, full_matrices=False)
    open_channel = vh.conj().T[:, 0]
    tau = singular_values**2

    n_prop_low = int(channels.low.N_prop)
    normal_index = normal_incidence_index(n_prop_low)
    v_low = np.zeros((n_prop_low, 2), dtype=np.complex128)
    v_low[normal_index, 0] = 1.0
    v_low[:, 1] = open_channel

    average_transmission = float(np.sum(np.abs(transmission) ** 2) / n_prop_low)
    plane_wave_transmission = float(np.sum(np.abs(transmission[:, normal_index]) ** 2))
    open_channel_transmission = float(tau[0])

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

    direct_field_profiles = None
    direct_info = None
    direct_field_difference_max = None
    if compute_direct_mesti_source:
        direct_field_profiles, direct_info = _direct_mesti_wavefront_field(
            epsilon,
            channels,
            v_low,
            wavelength=wavelength,
            dx=dx,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            yBC=yBC,
            pml_npixels=int(pml_npixels),
            solver=solver,
        )
        mesti2s_core, direct_core = _comparison_slices(
            field_profiles,
            direct_field_profiles,
            nz_low=int(nz_low),
            nz_high=int(nz_high),
            pml_npixels=int(pml_npixels),
        )
        direct_field_difference_max = float(np.max(np.abs(mesti2s_core - direct_core)))

    return OpenChannelResult(
        transmission=transmission,
        channels=channels,
        singular_values=singular_values,
        transmission_eigenvalues=tau,
        open_channel=open_channel,
        normal_index=normal_index,
        v_low=v_low,
        average_transmission=average_transmission,
        plane_wave_transmission=plane_wave_transmission,
        open_channel_transmission=open_channel_transmission,
        field_profiles=field_profiles,
        direct_field_profiles=direct_field_profiles,
        direct_field_difference_max=direct_field_difference_max,
        transmission_info=transmission_info,
        field_info=field_info,
        direct_info=direct_info,
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


def run_fixture_example(path: Path | None = None) -> OpenChannelResult:
    """Run the example against the reduced Julia-generated fixture."""

    fixture_path = (
        path
        if path is not None
        else Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "example_open_channel_through_disorder_v5.mat"
    )
    fixture = _load_fixture(fixture_path)
    return run_open_channel_through_disorder(
        epsilon_xx=np.asarray(fixture["epsilon_xx"], dtype=np.complex128),
        wavelength=float(_scalar(fixture, "wavelength")),
        dx=float(_scalar(fixture, "dx")),
        epsilon_low=_scalar(fixture, "epsilon_low"),
        epsilon_high=_scalar(fixture, "epsilon_high"),
        yBC=_fixture_string(fixture, "yBC"),
        pml_npixels=int(_scalar(fixture, "pml_npixels")),
        nz_low=int(_scalar(fixture, "nz_low")),
        nz_high=int(_scalar(fixture, "nz_high")),
        solver="scipy",
    )


def main() -> int:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "example_open_channel_through_disorder_v5.mat"
    )
    result = run_fixture_example(fixture_path) if fixture_path.exists() else run_open_channel_through_disorder()
    print(f"T_avg  = {result.average_transmission:.6f}")
    print(f"T_PW   = {result.plane_wave_transmission:.6f}")
    print(f"T_open = {result.open_channel_transmission:.6f}")
    if result.direct_field_difference_max is not None:
        print(f"max direct-field difference = {result.direct_field_difference_max:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
