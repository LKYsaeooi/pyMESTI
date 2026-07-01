"""Reduced metalens focusing example with angular spectrum propagation.

This is a compact Python translation of the numerical core from Julia
``examples/2d_metalens_focusing_via_angular_spectrum_propagation``. It builds
truncated incident plane-wave sources, samples the transmitted field just after
a tiny deterministic metalens with ``mesti``, and propagates that field to a
focal plane with angular spectrum propagation (ASP). Plotting, animation, and
the production-size metalens design file are left out of this runnable example.

Run from ``Simulation/python``:

    python examples/metalens_focusing_via_angular_spectrum_propagation.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesti import Opts, PML, Source_struct, Syst, asp as _mesti_asp, mesti, mesti_build_channels


@dataclass
class ASPSetup:
    """Sampling and wave-number data used by angular spectrum propagation."""

    dy_asp: float
    ny_asp: int
    w_asp: float
    ind_asp: np.ndarray
    ny_asp_pad_low: int
    ny_asp_pad_high: int
    y_asp: np.ndarray
    ky_asp: np.ndarray
    kx_asp: np.ndarray
    prop_indices: np.ndarray
    ky_asp_prop: np.ndarray
    kx_asp_prop: np.ndarray


@dataclass
class MetalensASPResult:
    """Numerical outputs from the reduced metalens ASP example."""

    epsilon_metalens: np.ndarray
    epsilon_syst: np.ndarray
    channels_left: Any
    theta_in_list: np.ndarray
    kydx_fov: np.ndarray
    kxdx_fov: np.ndarray
    y_source: np.ndarray
    ind_source_out: np.ndarray
    b_trunc: np.ndarray
    b_left: np.ndarray
    source: Source_struct
    projection: Source_struct
    c_right: np.ndarray
    asp_setup: ASPSetup
    focal_length: float
    focal_spot_list: np.ndarray
    field_right_after_metalens: np.ndarray
    field_at_focal_plane: np.ndarray
    focal_plane_intensity: np.ndarray
    target_focal_indices: np.ndarray
    target_focal_intensities: np.ndarray
    peak_intensities: np.ndarray
    peak_indices: np.ndarray
    peak_y_positions: np.ndarray
    direct_info: Any


def demo_metalens_permittivity(ny: int = 8, nz: int = 4, n_struct: float = 1.45) -> np.ndarray:
    """Build a tiny deterministic lens-like TM permittivity profile."""

    if ny <= 0 or nz <= 0:
        raise ValueError("ny and nz must be positive.")
    epsilon = np.ones((ny, nz), dtype=np.complex128)
    center = (ny + 1) / 2
    for row in range(ny):
        normalized_radius = abs((row + 1) - center) / (ny / 2)
        filled_cols = int(np.clip(round(nz * (0.28 + 0.52 * (1 - normalized_radius))), 1, nz))
        parity_offset = 0 if (row + 1) % 2 else 1
        for col in range(nz):
            if (col + 1) <= filled_cols or ((col + 1) + parity_offset) % 4 == 0:
                epsilon[row, col] = n_struct**2
    return epsilon


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive.")
    return 1 << (int(value) - 1).bit_length()


def angular_spectrum_propagation(
    f0: Any,
    x: Any,
    kx_prop: Any,
    ny_tot: int | None = None,
    ny_pad_low: int | None = None,
) -> np.ndarray:
    """Propagate a scalar profile with the Julia example's ASP convention."""

    return _mesti_asp(f0, x, kx_prop, ny_tot, ny_pad_low)


def build_truncated_sources(
    *,
    ny: int,
    ny_l: int,
    dx: float,
    d_in: float,
    k0dx: float,
    epsilon_l: complex,
    ybc_channels: str | float,
    theta_in_list: Any,
    use_continuous_dispersion: bool,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the truncated source profiles from the Julia metalens example."""

    theta = np.asarray(theta_in_list, dtype=float).reshape(-1)
    if theta.size == 0:
        raise ValueError("theta_in_list must contain at least one incident angle.")
    channels_l = mesti_build_channels(
        int(ny),
        ybc_channels,
        float(k0dx),
        epsilon_l,
        None,
        bool(use_continuous_dispersion),
    )
    kydx_fov = k0dx * np.sin(np.deg2rad(theta))
    kxdx_fov = np.sqrt((k0dx**2 - kydx_fov**2).astype(np.complex128))

    b_basis = channels_l.f_x_m(channels_l.kydx_prop)
    y_source = (np.arange(0.5, ny, 1.0) - ny / 2) * dx
    ind_source_out = np.flatnonzero(np.abs(y_source) > d_in / 2)
    b_trunc = channels_l.f_x_m(kydx_fov) * np.sqrt(ny / ny_l)
    b_trunc[ind_source_out, :] = 0

    coefficients = b_basis.conj().T @ b_trunc
    source_weights = (
        np.asarray(channels_l.sqrt_nu_prop, dtype=np.complex128)
        * np.exp((-0.5j) * np.asarray(channels_l.kzdx_prop, dtype=np.complex128))
    ).reshape(-1, 1)
    b_left = b_basis @ (source_weights * coefficients)

    return channels_l, kydx_fov, kxdx_fov, y_source, ind_source_out, b_trunc, b_left


def build_asp_setup(
    *,
    dx: float,
    wavelength: float,
    n_air: float,
    d_out: float,
    ny_r: int,
    dy_asp: float,
) -> ASPSetup:
    """Build the reduced ASP sampling grid and wave-number list."""

    ratio = dy_asp / dx
    rounded_ratio = round(ratio)
    if rounded_ratio != ratio:
        ratio = max(1, rounded_ratio)
        dy_asp = ratio * dx
    if ny_r % 2 == 0 and int(round(ratio)) % 2 == 0:
        ratio = ratio - 1
        dy_asp = ratio * dx

    w_asp_min = 2 * d_out
    ny_asp = _next_power_of_two(int(round(w_asp_min / dy_asp)))
    w_asp = ny_asp * dy_asp
    ind_asp_one_based = np.rint(np.arange(1, ny_r + 1e-12, ratio)).astype(int)
    if ind_asp_one_based[-1] != ny_r:
        if (ny_r - ind_asp_one_based[-1]) % 2 != 0:
            ind_asp_one_based = ind_asp_one_based[:-1]
        ind_asp_one_based = (
            ind_asp_one_based + ((ny_r - ind_asp_one_based[-1]) / 2)
        ).astype(int)
    ind_asp = ind_asp_one_based - 1

    ny_asp_pad = ny_asp - ind_asp.size
    ny_asp_pad_low = int(round(ny_asp_pad / 2))
    ny_asp_pad_high = ny_asp_pad - ny_asp_pad_low
    y_asp = (
        np.arange(0.5, ny_asp, 1.0)
        - 0.5 * (ny_asp + ny_asp_pad_low - ny_asp_pad_high)
    ) * dy_asp

    ny_asp_half = int(round(ny_asp / 2))
    ky_asp = (2 * np.pi / w_asp) * np.r_[np.arange(0, ny_asp_half), np.arange(-ny_asp_half, 0)]
    k0 = n_air * 2 * np.pi / wavelength
    kx_asp = np.sqrt((k0**2 - ky_asp**2).astype(np.complex128))
    prop_indices = np.flatnonzero(np.abs(ky_asp) < k0)
    ky_asp_prop = ky_asp[prop_indices]
    kx_asp_prop = np.sqrt((k0**2 - ky_asp_prop**2).astype(np.complex128))

    return ASPSetup(
        dy_asp=float(dy_asp),
        ny_asp=int(ny_asp),
        w_asp=float(w_asp),
        ind_asp=ind_asp.astype(int),
        ny_asp_pad_low=int(ny_asp_pad_low),
        ny_asp_pad_high=int(ny_asp_pad_high),
        y_asp=y_asp,
        ky_asp=ky_asp,
        kx_asp=kx_asp,
        prop_indices=prop_indices.astype(int),
        ky_asp_prop=ky_asp_prop,
        kx_asp_prop=kx_asp_prop,
    )


def run_metalens_focusing_via_angular_spectrum_propagation(
    *,
    epsilon_metalens: Any | None = None,
    n_air: float = 1.0,
    n_sub: float = 1.0,
    n_struct: float = 1.45,
    wavelength: float = 1.0,
    dx: float = 0.25,
    d_out: float = 2.0,
    d_in: float = 1.0,
    h: float = 1.0,
    numerical_aperture: float = 0.6,
    w_out: float = 3.0,
    theta_in_list: Any = (-20.0, 0.0, 20.0),
    pml_npixels: int = 2,
    dy_asp: float | None = None,
    ybc_channels: str | float = "periodic",
    use_continuous_dispersion: bool = True,
    solver: str | None = "scipy",
) -> MetalensASPResult:
    """Run the reduced metalens focusing and ASP example."""

    if wavelength <= 0 or dx <= 0:
        raise ValueError("wavelength and dx must be positive.")
    if d_out <= 0 or d_in <= 0 or h <= 0 or w_out < d_out:
        raise ValueError("d_out, d_in, h, and w_out must describe a positive aperture.")
    if pml_npixels < 0:
        raise ValueError("pml_npixels must be nonnegative.")
    if numerical_aperture <= 0 or numerical_aperture >= n_air:
        raise ValueError("numerical_aperture must be positive and smaller than n_air.")

    ny_r_extra_half = int(round((w_out - d_out) / dx / 2))
    ny = int(np.ceil(d_out / dx))
    ny_l = int(np.ceil(d_in / dx))
    ny_r = ny + 2 * ny_r_extra_half
    nz = int(np.ceil(h / dx))

    epsilon_l = n_sub**2
    epsilon_r = n_air**2
    epsilon_lens = (
        demo_metalens_permittivity(ny, nz, n_struct)
        if epsilon_metalens is None
        else np.asarray(epsilon_metalens, dtype=np.complex128)
    )
    if epsilon_lens.shape != (ny, nz):
        raise ValueError(f"epsilon_metalens must have shape {(ny, nz)}.")

    nz_extra_left = 1 + int(pml_npixels)
    nz_extra_right = nz_extra_left
    ny_extra_low = ny_r_extra_half + int(pml_npixels)
    ny_extra_high = ny_extra_low
    ny_tot = ny + ny_extra_low + ny_extra_high
    nz_tot = nz + nz_extra_left + nz_extra_right

    k0dx = (2 * np.pi / wavelength) * dx
    (
        channels_l,
        kydx_fov,
        kxdx_fov,
        y_source,
        ind_source_out,
        b_trunc,
        b_left,
    ) = build_truncated_sources(
        ny=ny,
        ny_l=ny_l,
        dx=dx,
        d_in=d_in,
        k0dx=k0dx,
        epsilon_l=epsilon_l,
        ybc_channels=ybc_channels,
        theta_in_list=theta_in_list,
        use_continuous_dispersion=use_continuous_dispersion,
    )

    epsilon_syst = np.ones((ny_tot, nz_tot), dtype=np.complex128)
    epsilon_syst[
        ny_extra_low : ny_extra_low + ny,
        nz_extra_left : nz_extra_left + nz,
    ] = epsilon_lens

    n_l = nz_extra_left - 1
    m1_l = ny_extra_low
    source = Source_struct(
        pos=[np.array([m1_l, n_l, m1_l + ny - 1, n_l], dtype=int)],
        data=[b_left],
    )

    asp_setup = build_asp_setup(
        dx=dx,
        wavelength=wavelength,
        n_air=n_air,
        d_out=d_out,
        ny_r=ny_r,
        dy_asp=dx if dy_asp is None else dy_asp,
    )

    n_r = n_l + nz + 1
    m1_r = int(pml_npixels)
    c_right = np.zeros((ny_r, asp_setup.ind_asp.size), dtype=np.complex128)
    c_right[asp_setup.ind_asp, np.arange(asp_setup.ind_asp.size)] = 1
    projection = Source_struct(
        pos=[np.array([m1_r, n_r, m1_r + ny_r - 1, n_r], dtype=int)],
        data=[c_right],
    )

    syst = Syst(
        epsilon_xx=epsilon_syst,
        wavelength=float(wavelength),
        dx=float(dx),
        PML=[PML(int(pml_npixels), direction="all")],
    )
    field_right_after_metalens, direct_info = mesti(
        syst,
        [source],
        [projection],
        opts=Opts(
            solver=solver,
            verbal=False,
            prefactor=-2j,
            use_L0_threads=False,
        ),
    )

    focal_length = d_out / 2 / np.tan(np.arcsin(numerical_aperture / n_air))
    theta = np.asarray(theta_in_list, dtype=float).reshape(-1)
    field_at_focal_plane = np.zeros((asp_setup.ny_asp, theta.size), dtype=np.complex128)
    for angle_index in range(theta.size):
        field_at_focal_plane[:, angle_index] = angular_spectrum_propagation(
            field_right_after_metalens[:, angle_index],
            focal_length,
            asp_setup.kx_asp_prop,
            asp_setup.ny_asp,
        )

    focal_spot_list = focal_length * np.tan(np.deg2rad(theta))
    target_focal_indices = np.array(
        [int(np.argmin(np.abs(asp_setup.y_asp - spot))) for spot in focal_spot_list],
        dtype=int,
    )
    focal_plane_intensity = np.abs(field_at_focal_plane) ** 2
    target_focal_intensities = focal_plane_intensity[
        target_focal_indices,
        np.arange(theta.size),
    ]
    peak_indices = np.argmax(focal_plane_intensity, axis=0).astype(int)
    peak_intensities = focal_plane_intensity[peak_indices, np.arange(theta.size)]
    peak_y_positions = asp_setup.y_asp[peak_indices]

    return MetalensASPResult(
        epsilon_metalens=epsilon_lens,
        epsilon_syst=epsilon_syst,
        channels_left=channels_l,
        theta_in_list=theta,
        kydx_fov=kydx_fov,
        kxdx_fov=kxdx_fov,
        y_source=y_source,
        ind_source_out=ind_source_out,
        b_trunc=b_trunc,
        b_left=b_left,
        source=source,
        projection=projection,
        c_right=c_right,
        asp_setup=asp_setup,
        focal_length=float(focal_length),
        focal_spot_list=focal_spot_list,
        field_right_after_metalens=field_right_after_metalens,
        field_at_focal_plane=field_at_focal_plane,
        focal_plane_intensity=focal_plane_intensity,
        target_focal_indices=target_focal_indices,
        target_focal_intensities=target_focal_intensities,
        peak_intensities=peak_intensities,
        peak_indices=peak_indices,
        peak_y_positions=peak_y_positions,
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


def _vector(data: dict[str, Any], key: str, dtype: Any | None = None) -> np.ndarray:
    return np.asarray(data[key], dtype=dtype).reshape(-1)


def _fixture_string(data: dict[str, Any], key: str) -> str:
    value = np.asarray(data[key])
    if value.dtype.kind in {"U", "S"}:
        return "".join(value.reshape(-1).astype(str)).strip()
    return str(_scalar(data, key))


def run_fixture_example(path: Path | None = None) -> MetalensASPResult:
    """Run the example against the reduced Julia-generated fixture."""

    fixture_path = (
        path
        if path is not None
        else Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "example_metalens_asp_v5.mat"
    )
    fixture = _load_fixture(fixture_path)
    return run_metalens_focusing_via_angular_spectrum_propagation(
        epsilon_metalens=np.asarray(fixture["epsilon_metalens"], dtype=np.complex128),
        n_air=float(_scalar(fixture, "n_air")),
        n_sub=float(_scalar(fixture, "n_sub")),
        n_struct=float(_scalar(fixture, "n_struct")),
        wavelength=float(_scalar(fixture, "wavelength")),
        dx=float(_scalar(fixture, "dx")),
        d_out=float(_scalar(fixture, "D_out")),
        d_in=float(_scalar(fixture, "D_in")),
        h=float(_scalar(fixture, "h")),
        numerical_aperture=float(_scalar(fixture, "NA")),
        w_out=float(_scalar(fixture, "W_out")),
        theta_in_list=_vector(fixture, "theta_in_list", dtype=float),
        pml_npixels=int(_scalar(fixture, "nPML")),
        dy_asp=float(_scalar(fixture, "dy_ASP_input")),
        ybc_channels=_fixture_string(fixture, "yBC_channels"),
        use_continuous_dispersion=bool(_scalar(fixture, "use_continuous_dispersion")),
        solver="scipy",
    )


def main() -> int:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "example_metalens_asp_v5.mat"
    )
    result = run_fixture_example(fixture_path) if fixture_path.exists() else run_metalens_focusing_via_angular_spectrum_propagation()
    print(f"right-after-metalens field shape = {result.field_right_after_metalens.shape}")
    print(f"focal-plane field shape = {result.field_at_focal_plane.shape}")
    print(f"target focal intensities = {result.target_focal_intensities}")
    print(f"peak focal intensities = {result.peak_intensities}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
