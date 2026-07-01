"""Reduced 3D open-channel-through-disorder MESTI example.

This is a compact Python translation of the numerical core from Julia
``examples/3d_open_channel_through_disorder``. It computes a low-to-high
both-polarization transmission matrix with ``mesti2s``, extracts closed and
open incident channels with SVD, and computes 3D ``Ex``, ``Ey``, and ``Ez``
field profiles for closed-channel, open-channel, and normal p-polarized
plane-wave inputs. Plotting, random Ball-smoothed disorder generation, and the
production-size out-of-core MUMPS workflow are left out of this runnable
example.

Run from ``Simulation/python``:

    python examples/open_channel_through_disorder_3d.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesti import Opts, PML, Syst, channel_type, mesti2s, wavefront


@dataclass
class OpenChannel3DResult:
    """Numerical outputs from the reduced 3D open-channel example."""

    epsilon_xx: np.ndarray
    epsilon_yy: np.ndarray
    epsilon_zz: np.ndarray
    transmission: np.ndarray
    channels: Any
    singular_values: np.ndarray
    transmission_eigenvalues: np.ndarray
    open_channel: np.ndarray
    closed_channel: np.ndarray
    normal_index: int
    normal_s_column: int
    normal_p_column: int
    v_low_s: np.ndarray
    v_low_p: np.ndarray
    average_transmission: float
    plane_wave_s_transmission: float
    plane_wave_p_transmission: float
    closed_channel_transmission: float
    open_channel_transmission: float
    field_Ex: np.ndarray
    field_Ey: np.ndarray
    field_Ez: np.ndarray
    combined_closed_Ex: np.ndarray
    combined_closed_Ey: np.ndarray
    combined_closed_Ez: np.ndarray
    combined_open_Ex: np.ndarray
    combined_open_Ey: np.ndarray
    combined_open_Ez: np.ndarray
    normal_p_Ex: np.ndarray
    normal_p_Ey: np.ndarray
    normal_p_Ez: np.ndarray
    normalized_closed_Ex: np.ndarray
    normalized_open_Ex: np.ndarray
    normalized_normal_p_Ex: np.ndarray
    open_ex_normalization_factor: float
    transmission_info: Any
    field_info: Any


def _patterned_epsilon(nx: int, ny: int, nz: int, base: float, loss: float = 0.05) -> np.ndarray:
    ix, iy, iz = np.indices((nx, ny, nz), dtype=float)
    ix = ix + 1.0
    iy = iy + 1.0
    iz = iz + 1.0
    epsilon = (
        base
        + 0.041 * ix
        + 0.026 * iy
        + 0.013 * iz
        + 0.007 * np.sin(1.7 * ix + 0.3 * iy + 0.5 * iz)
        + 1j * loss
    )
    return epsilon.astype(np.complex128)


def demo_vectorial_disorder_permittivity(
    nx: int = 3,
    ny: int = 3,
    nz: int = 1,
    loss: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a tiny deterministic diagonal 3D tensor permittivity."""

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("nx, ny, and nz must be positive.")
    epsilon_xx = _patterned_epsilon(nx, ny, nz, 1.04, loss)
    epsilon_yy = _patterned_epsilon(nx, ny, nz, 1.12, loss)
    epsilon_zz = _patterned_epsilon(nx, ny, nz + 1, 1.20, loss)
    return epsilon_xx, epsilon_yy, epsilon_zz


def normal_incidence_index_3d(n_prop: int) -> int:
    """Return the zero-based normal-incidence channel index used by Julia."""

    if n_prop <= 0:
        raise ValueError("At least one propagating channel is required.")
    return int(round((n_prop + 1) / 2)) - 1


def _system(
    epsilon_xx: np.ndarray,
    epsilon_yy: np.ndarray,
    epsilon_zz: np.ndarray,
    *,
    wavelength: float,
    dx: float,
    epsilon_low: complex,
    epsilon_high: complex,
    xBC: str | float,
    yBC: str | float,
    pml_npixels: int,
) -> Syst:
    return Syst(
        epsilon_xx=np.asarray(epsilon_xx, dtype=np.complex128),
        epsilon_yy=np.asarray(epsilon_yy, dtype=np.complex128),
        epsilon_zz=np.asarray(epsilon_zz, dtype=np.complex128),
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        wavelength=float(wavelength),
        dx=float(dx),
        xBC=xBC,
        yBC=yBC,
        zPML=[PML(int(pml_npixels))],
    )


def _validate_permittivity_shapes(
    epsilon_xx: np.ndarray,
    epsilon_yy: np.ndarray,
    epsilon_zz: np.ndarray,
) -> None:
    if epsilon_xx.ndim != 3 or epsilon_yy.ndim != 3 or epsilon_zz.ndim != 3:
        raise ValueError("epsilon_xx, epsilon_yy, and epsilon_zz must be 3D arrays.")
    if epsilon_xx.shape != epsilon_yy.shape:
        raise ValueError("This reduced example expects epsilon_xx and epsilon_yy to share a shape.")
    if epsilon_zz.shape[:2] != epsilon_xx.shape[:2] or epsilon_zz.shape[2] != epsilon_xx.shape[2] + 1:
        raise ValueError(
            "This reduced example expects epsilon_zz shape (nx, ny, nz + 1) "
            "for the same transverse grid as epsilon_xx."
        )


def _split_svd_wavefronts(
    transmission: np.ndarray,
    n_prop_low: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    _, singular_values, vh = np.linalg.svd(transmission, full_matrices=False)
    right_vectors = vh.conj().T
    open_channel = right_vectors[:, 0]
    closed_channel = right_vectors[:, -1]

    normal_index = normal_incidence_index_3d(n_prop_low)
    normal_s_column = normal_index
    normal_p_column = n_prop_low + normal_index

    v_low_s = np.zeros((n_prop_low, 2), dtype=np.complex128)
    v_low_p = np.zeros((n_prop_low, 3), dtype=np.complex128)
    v_low_s[:, 0] = closed_channel[:n_prop_low]
    v_low_p[:, 0] = closed_channel[n_prop_low : 2 * n_prop_low]
    v_low_s[:, 1] = open_channel[:n_prop_low]
    v_low_p[:, 1] = open_channel[n_prop_low : 2 * n_prop_low]
    v_low_p[normal_index, 2] = 1.0

    return (
        singular_values,
        open_channel,
        closed_channel,
        v_low_s,
        v_low_p,
        normal_index,
        normal_s_column,
        normal_p_column,
    )


def _combine_field_columns(
    field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if field.shape[3] != 5:
        raise RuntimeError(f"Expected five 3D wavefront field columns; got {field.shape[3]}.")
    closed = field[:, :, :, 0] + field[:, :, :, 2]
    opened = field[:, :, :, 1] + field[:, :, :, 3]
    normal_p = field[:, :, :, 4]
    return closed, opened, normal_p


def run_open_channel_through_disorder_3d(
    *,
    epsilon_xx: Any | None = None,
    epsilon_yy: Any | None = None,
    epsilon_zz: Any | None = None,
    wavelength: float = 2 * np.pi / 1.8,
    dx: float = 1.0,
    epsilon_low: complex = 1.0,
    epsilon_high: complex = 1.0,
    xBC: str | float = "periodic",
    yBC: str | float = "periodic",
    pml_npixels: int = 16,
    solver: str | None = "scipy",
) -> OpenChannel3DResult:
    """Run the reduced 3D open-channel example and return numerical results."""

    if epsilon_xx is None and epsilon_yy is None and epsilon_zz is None:
        eps_xx, eps_yy, eps_zz = demo_vectorial_disorder_permittivity()
    elif epsilon_xx is None or epsilon_yy is None or epsilon_zz is None:
        raise ValueError("epsilon_xx, epsilon_yy, and epsilon_zz must be provided together.")
    else:
        eps_xx = np.asarray(epsilon_xx, dtype=np.complex128)
        eps_yy = np.asarray(epsilon_yy, dtype=np.complex128)
        eps_zz = np.asarray(epsilon_zz, dtype=np.complex128)
    _validate_permittivity_shapes(eps_xx, eps_yy, eps_zz)
    if pml_npixels < 0:
        raise ValueError("pml_npixels must be nonnegative.")

    syst = _system(
        eps_xx,
        eps_yy,
        eps_zz,
        wavelength=wavelength,
        dx=dx,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        xBC=xBC,
        yBC=yBC,
        pml_npixels=pml_npixels,
    )

    transmission, channels, transmission_info = mesti2s(
        syst,
        channel_type(side="low", polarization="both"),
        channel_type(side="high", polarization="both"),
        Opts(solver=solver, verbal=False, use_L0_threads=False),
    )
    n_prop_low = int(channels.low.N_prop)
    (
        singular_values,
        open_channel,
        closed_channel,
        v_low_s,
        v_low_p,
        normal_index,
        normal_s_column,
        normal_p_column,
    ) = _split_svd_wavefronts(transmission, n_prop_low)
    tau = singular_values**2

    average_transmission = float(np.sum(np.abs(transmission) ** 2) / (2 * n_prop_low))
    plane_wave_s_transmission = float(np.sum(np.abs(transmission[:, normal_s_column]) ** 2))
    plane_wave_p_transmission = float(np.sum(np.abs(transmission[:, normal_p_column]) ** 2))
    closed_channel_transmission = float(tau[-1])
    open_channel_transmission = float(tau[0])

    field_Ex, field_Ey, field_Ez, _, field_info = mesti2s(
        syst,
        wavefront(v_low_s=v_low_s, v_low_p=v_low_p),
        Opts(solver=solver, verbal=False, use_L0_threads=False),
    )
    combined_closed_Ex, combined_open_Ex, normal_p_Ex = _combine_field_columns(field_Ex)
    combined_closed_Ey, combined_open_Ey, normal_p_Ey = _combine_field_columns(field_Ey)
    combined_closed_Ez, combined_open_Ez, normal_p_Ez = _combine_field_columns(field_Ez)

    normalization_factor = float(np.max(np.abs(combined_open_Ex)))
    if normalization_factor == 0:
        raise RuntimeError("Open-channel Ex field has zero maximum amplitude.")

    return OpenChannel3DResult(
        epsilon_xx=eps_xx,
        epsilon_yy=eps_yy,
        epsilon_zz=eps_zz,
        transmission=transmission,
        channels=channels,
        singular_values=singular_values,
        transmission_eigenvalues=tau,
        open_channel=open_channel,
        closed_channel=closed_channel,
        normal_index=normal_index,
        normal_s_column=normal_s_column,
        normal_p_column=normal_p_column,
        v_low_s=v_low_s,
        v_low_p=v_low_p,
        average_transmission=average_transmission,
        plane_wave_s_transmission=plane_wave_s_transmission,
        plane_wave_p_transmission=plane_wave_p_transmission,
        closed_channel_transmission=closed_channel_transmission,
        open_channel_transmission=open_channel_transmission,
        field_Ex=field_Ex,
        field_Ey=field_Ey,
        field_Ez=field_Ez,
        combined_closed_Ex=combined_closed_Ex,
        combined_closed_Ey=combined_closed_Ey,
        combined_closed_Ez=combined_closed_Ez,
        combined_open_Ex=combined_open_Ex,
        combined_open_Ey=combined_open_Ey,
        combined_open_Ez=combined_open_Ez,
        normal_p_Ex=normal_p_Ex,
        normal_p_Ey=normal_p_Ey,
        normal_p_Ez=normal_p_Ez,
        normalized_closed_Ex=combined_closed_Ex / normalization_factor,
        normalized_open_Ex=combined_open_Ex / normalization_factor,
        normalized_normal_p_Ex=normal_p_Ex / normalization_factor,
        open_ex_normalization_factor=normalization_factor,
        transmission_info=transmission_info,
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
    value = np.asarray(data[key])
    if value.dtype.kind in {"U", "S"}:
        return "".join(value.reshape(-1).astype(str)).strip()
    return str(_scalar(data, key))


def run_fixture_example(path: Path | None = None) -> OpenChannel3DResult:
    """Run the example against the reduced Julia-generated fixture."""

    fixture_path = (
        path
        if path is not None
        else Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "example_3d_open_channel_through_disorder_v5.mat"
    )
    fixture = _load_fixture(fixture_path)
    return run_open_channel_through_disorder_3d(
        epsilon_xx=np.asarray(fixture["epsilon_xx"], dtype=np.complex128),
        epsilon_yy=np.asarray(fixture["epsilon_yy"], dtype=np.complex128),
        epsilon_zz=np.asarray(fixture["epsilon_zz"], dtype=np.complex128),
        wavelength=float(_scalar(fixture, "wavelength")),
        dx=float(_scalar(fixture, "dx")),
        epsilon_low=_scalar(fixture, "epsilon_low"),
        epsilon_high=_scalar(fixture, "epsilon_high"),
        xBC=_fixture_string(fixture, "xBC"),
        yBC=_fixture_string(fixture, "yBC"),
        pml_npixels=int(_scalar(fixture, "zPML_npixels")),
        solver="scipy",
    )


def main() -> int:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "example_3d_open_channel_through_disorder_v5.mat"
    )
    result = (
        run_fixture_example(fixture_path)
        if fixture_path.exists()
        else run_open_channel_through_disorder_3d()
    )
    print(f"T_avg     = {result.average_transmission:.6f}")
    print(f"T_PW_s    = {result.plane_wave_s_transmission:.6f}")
    print(f"T_PW_p    = {result.plane_wave_p_transmission:.6f}")
    print(f"T_closed  = {result.closed_channel_transmission:.6f}")
    print(f"T_open    = {result.open_channel_transmission:.6f}")
    print(
        "field shapes = "
        f"Ex{result.field_Ex.shape}, Ey{result.field_Ey.shape}, Ez{result.field_Ez.shape}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
