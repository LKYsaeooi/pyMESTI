"""Translated MESTI example helpers.

The functions in this module keep the numerical core of selected Julia
``examples/`` scripts runnable from Python while avoiding plotting and notebook
dependencies.  They are intentionally small and fixture-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from .channels import mesti_build_channels
from .mesti import mesti
from .mumps import (
    Mumps,
    get_schur_complement,
    mumps_schur_complement_inplace,
    mumps_solve,
    set_icntl,
    set_job,
    set_schur_centralized_by_column,
)
from .types import Opts, PML, Source_struct, Syst


@dataclass
class GaussianBeamProfiles:
    """Line-source and projection profiles for the Gaussian reflection example."""

    B_low: np.ndarray
    C_low: np.ndarray
    y_focus: np.ndarray
    beam_radius: float
    channels: Any
    transpose_mismatch_max: float


@dataclass
class GaussianReflectionResult:
    """Numerical outputs from the reduced Gaussian-beam reflection example."""

    source: Source_struct
    profiles: GaussianBeamProfiles
    reference: np.ndarray
    reflection: np.ndarray
    field_profiles: np.ndarray | None
    reference_info: Any
    reflection_info: Any
    field_info: Any | None


@dataclass
class BasicMumpsSolveDemoResult:
    """Outputs from the compact raw-MUMPS basic-solve demo translation."""

    matrix: sparse.csc_matrix
    rhs: np.ndarray | sparse.csc_matrix
    solution: np.ndarray
    residual_norm: float
    residual_tolerance: float
    dtype: np.dtype
    sparse_rhs: bool


@dataclass
class MumpsSchurComplementDemoResult:
    """Outputs from the compact raw-MUMPS Schur-complement demo translation."""

    matrix: sparse.csc_matrix
    schur_indices: np.ndarray
    schur: np.ndarray
    expected_schur: np.ndarray
    relative_error: float
    relative_tolerance: float
    dtype: np.dtype


@dataclass
class TransmissionDistributionComparison:
    """Histogram data matching the open-channel DMPK comparison helper."""

    bin_edges: np.ndarray
    bin_centers: np.ndarray
    counts: np.ndarray
    pdf: np.ndarray
    dmpk_pdf: np.ndarray
    mean_tau: float
    bin_width: float


def _demo_complex_dtype(dtype: Any) -> np.dtype:
    result = np.dtype(dtype)
    if result in {np.dtype(np.float32), np.dtype(np.complex64)}:
        return np.dtype(np.complex64)
    if result in {np.dtype(np.float64), np.dtype(np.complex128)}:
        return np.dtype(np.complex128)
    raise ValueError("dtype must be float32, float64, complex64, or complex128.")


def _demo_real_dtype(dtype: np.dtype) -> np.dtype:
    return np.dtype(np.float32 if dtype == np.dtype(np.complex64) else np.float64)


def asp(
    f0: Any,
    x: Any,
    kx_prop: Any,
    ny_tot: int | None = None,
    ny_pad_low: int | None = None,
) -> np.ndarray:
    """Propagate a scalar field with Julia ``asp.jl`` conventions."""

    field = np.asarray(f0, dtype=np.complex128)
    input_was_vector = field.ndim == 1
    if field.ndim == 1:
        field = field.reshape(-1, 1)
    if field.ndim != 2:
        raise ValueError("f0 must be a vector or 2D matrix.")

    x_arr = np.asarray(x)
    x_is_scalar = x_arr.ndim == 0 or x_arr.size == 1
    if x_arr.ndim == 2 and x_arr.shape[0] != 1:
        raise ValueError("x must be a scalar or row-vector equivalent.")
    if x_arr.ndim > 2:
        raise ValueError("x must be a scalar or row-vector equivalent.")
    if field.shape[1] > 1 and not x_is_scalar:
        raise ValueError("x must be a scalar when f0 has more than one column.")
    x_values = x_arr.reshape(-1)

    ny = field.shape[0]
    kx = np.asarray(kx_prop, dtype=np.complex128).reshape(-1)
    n_prop = kx.size
    if ny_tot is None:
        ny_tot = ny
    ny_tot = int(ny_tot)
    if ny_tot < ny:
        raise ValueError(f"ny_tot, when given, must be no smaller than size(f0,1) = {ny}.")
    if ny_tot < n_prop:
        raise ValueError(f"ny_tot, when given, must be no smaller than length(kx_prop) = {n_prop}.")
    if ny_pad_low is None:
        ny_pad_low = int(round((ny_tot - ny) / 2))
    ny_pad_low = int(ny_pad_low)
    if ny_pad_low + ny > ny_tot:
        raise ValueError(f"ny_pad_low + ny must be no greater than size(f0,1) = {ny}.")

    padded = np.vstack([field, np.zeros((ny_tot - ny, field.shape[1]), dtype=np.complex128)])
    pre_phase = np.exp((-2j * np.pi * ny_pad_low / ny_tot) * np.arange(ny_tot)).reshape(-1, 1)
    field_fft = pre_phase * np.fft.fft(padded, axis=0)

    if x_is_scalar:
        phase = np.exp(1j * kx.reshape(-1, 1) * x_values[0])
    else:
        phase = np.exp(1j * kx.reshape(-1, 1) * x_values.reshape(1, -1))

    if n_prop == ny_tot:
        propagated = np.fft.ifft(phase * field_fft, axis=0)
    else:
        if n_prop % 2 != 1:
            raise ValueError(f"length(kx_prop) = {n_prop} must be an odd number when it is not ny_tot.")
        a_max = int(round((n_prop - 1) / 2))
        prop_indices = np.r_[np.arange(a_max + 1), np.arange(ny_tot - a_max, ny_tot)]
        propagated_subset = phase * field_fft[prop_indices, :]
        shifted = np.roll(propagated_subset, a_max, axis=0)
        padded_subset = np.vstack(
            [
                shifted,
                np.zeros((ny_tot - n_prop, shifted.shape[1]), dtype=np.complex128),
            ]
        )
        post_phase = np.exp((-2j * np.pi * a_max / ny_tot) * np.arange(ny_tot)).reshape(-1, 1)
        propagated = post_phase * np.fft.ifft(padded_subset, axis=0)

    if input_was_vector and x_is_scalar:
        return propagated.reshape(-1)
    return propagated


def plot_and_compare_distribution(tau: Any, *, bin_width: float = 0.02) -> TransmissionDistributionComparison:
    """Return the histogram/DMPK data from Julia ``plot_and_compare_distribution``.

    The Julia helper renders a plot.  The Python port keeps the numerical data
    reusable and leaves rendering to caller-selected plotting tools.
    """

    tau_arr = np.asarray(tau, dtype=float).reshape(-1)
    if tau_arr.size == 0:
        raise ValueError("tau must contain at least one transmission eigenvalue.")
    if np.any((tau_arr < 0) | (tau_arr > 1)):
        raise ValueError("tau values must lie in [0, 1].")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")
    n_bins_float = 1.0 / float(bin_width)
    n_bins = int(round(n_bins_float))
    if not np.isclose(n_bins_float, n_bins, rtol=0, atol=1e-12):
        raise ValueError("bin_width must divide the [0, 1] interval evenly.")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(tau_arr, bins=bin_edges)
    bin_width = float(bin_edges[1] - bin_edges[0])
    pdf = counts.astype(float) / (float(np.sum(counts)) * bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_tau = float(np.mean(tau_arr))
    dmpk_pdf = mean_tau / (2 * bin_centers * np.sqrt(1 - bin_centers))

    return TransmissionDistributionComparison(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        counts=counts,
        pdf=pdf,
        dmpk_pdf=dmpk_pdf,
        mean_tau=mean_tau,
        bin_width=bin_width,
    )


def build_epsilon_disorder(
    W: float,
    L: float,
    r_min: float,
    r_max: float,
    min_sep: float,
    number_density: float,
    rng_seed: int,
    dx: float,
    epsilon_scat: complex,
    epsilon_bg: complex,
    build_TM: bool,
    build_TE: bool = False,
    yBC: str = "periodic",
    y1: float = 0,
    y2: float | None = None,
    z1: float = 0,
    z2: float | None = None,
    *,
    no_scatterer_center: bool = False,
) -> tuple[Any, ...]:
    """Compatibility stub for Julia's 2D random Ball-disorder builder."""

    _ = (r_min, r_max, min_sep, number_density, rng_seed, epsilon_scat, epsilon_bg, yBC, no_scatterer_center)
    if dx <= 0:
        raise ValueError("dx must be positive.")
    if y2 is None:
        y2 = W
    if z2 is None:
        z2 = L
    if y1 < 0 or y2 > W or z1 < 0 or z2 > L:
        raise ValueError("The ranges [y1, y2] and [z1, z2] are invalid.")
    if not build_TM and not build_TE:
        raise ValueError("At least one of build_TM or build_TE must be true.")
    raise NotImplementedError(
        "build_epsilon_disorder requires Julia Ball subpixel smoothing. "
        "The Python port exposes Ball as an explicit unsupported stub; use a "
        "Julia-generated epsilon fixture or pass epsilon arrays directly to the reduced examples."
    )


def build_epsilon_disorder_3d(
    W_x: float,
    W_y: float,
    L: float,
    r_min: float,
    r_max: float,
    min_sep: float,
    number_density: float,
    rng_seed: int,
    dx: float,
    epsilon_scat: complex,
    epsilon_bg: complex,
    x1: float = 0,
    x2: float | None = None,
    y1: float = 0,
    y2: float | None = None,
    z1: float = 0,
    z2: float | None = None,
    *,
    no_scatterer_center: bool = False,
) -> tuple[Any, ...]:
    """Compatibility stub for Julia's 3D random Ball-disorder builder."""

    _ = (r_min, r_max, min_sep, number_density, rng_seed, epsilon_scat, epsilon_bg, no_scatterer_center)
    if dx <= 0:
        raise ValueError("dx must be positive.")
    if x2 is None:
        x2 = W_x
    if y2 is None:
        y2 = W_y
    if z2 is None:
        z2 = L
    if x1 < 0 or x2 > W_x or y1 < 0 or y2 > W_y or z1 < 0 or z2 > L:
        raise ValueError("The ranges [x1, x2], [y1, y2], and [z1, z2] are invalid.")
    raise NotImplementedError(
        "build_epsilon_disorder_3d requires Julia Ball subpixel smoothing. "
        "The Python port exposes Ball as an explicit unsupported stub; use a "
        "Julia-generated epsilon fixture or pass epsilon tensors directly to the reduced examples."
    )


def basic_mumps_solve_demo(
    *,
    n: int = 12,
    nrhs: int = 3,
    dtype: Any = np.complex128,
    sparse_rhs: bool = False,
) -> BasicMumpsSolveDemoResult:
    """Run a compact Python analogue of Julia ``mumps/basic_solve.jl``.

    The Julia script stress-tests random sparse systems through raw MUMPS.  This
    deterministic Python helper keeps the user-facing solve surface runnable
    through the SciPy-backed compatibility facade without requiring MPI.
    """

    if n <= 1:
        raise ValueError("n must be greater than 1.")
    if nrhs <= 0:
        raise ValueError("nrhs must be positive.")

    complex_dtype = _demo_complex_dtype(dtype)
    real_dtype = _demo_real_dtype(complex_dtype)
    main_diag = 2.5 + 0.05 * np.arange(n)
    lower_diag = -0.18 + 0.01j * np.arange(1, n)
    upper_diag = 0.11 - 0.015j * np.arange(1, n)
    matrix = sparse.diags(
        (lower_diag, main_diag, upper_diag),
        offsets=(-1, 0, 1),
        shape=(n, n),
        format="csc",
        dtype=complex_dtype,
    )

    rows = np.arange(n, dtype=float)[:, np.newaxis]
    cols = np.arange(nrhs, dtype=float)[np.newaxis, :]
    rhs_dense = ((rows + 1.0) + 1j * (cols + 1.0)) / (n + nrhs)
    rhs_dense[(rows.astype(int) + 2 * cols.astype(int)) % 3 == 0] = 0.0
    rhs_dense = rhs_dense.astype(complex_dtype)
    rhs = sparse.csc_matrix(rhs_dense) if sparse_rhs else rhs_dense

    solution = mumps_solve(matrix, rhs)
    residual = matrix.astype(np.complex128) @ solution - rhs_dense.astype(np.complex128)
    residual_norm = float(np.linalg.norm(residual))
    residual_tolerance = float(np.sqrt(np.finfo(real_dtype).eps) * max(1.0, np.linalg.norm(rhs_dense)))

    return BasicMumpsSolveDemoResult(
        matrix=matrix,
        rhs=rhs,
        solution=solution,
        residual_norm=residual_norm,
        residual_tolerance=residual_tolerance,
        dtype=complex_dtype,
        sparse_rhs=bool(sparse_rhs),
    )


def mumps_schur_complement_demo(
    *,
    m: int = 6,
    n: int = 3,
    dtype: Any = np.complex128,
) -> MumpsSchurComplementDemoResult:
    """Run a compact Python analogue of Julia ``mumps/schur_complement.jl``.

    Python Schur selectors are zero-based, so the returned indices correspond
    to Julia's trailing one-based ``m+1:m+n`` block.  Raw ``invoke_mumps`` jobs
    remain unsupported; this helper uses the validated compatibility algebra.
    """

    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive.")

    complex_dtype = _demo_complex_dtype(dtype)
    real_dtype = _demo_real_dtype(complex_dtype)
    row_m, col_m = np.indices((m, m), dtype=float)
    row_mn, col_mn = np.indices((m, n), dtype=float)
    row_nm, col_nm = np.indices((n, m), dtype=float)
    row_n, col_n = np.indices((n, n), dtype=float)

    A = np.diag(2.8 + 0.2 * np.arange(m)).astype(np.complex128)
    A += np.triu(0.025 * (row_m + 1) / (col_m + 2) + 0.006j * (row_m - col_m), k=1)
    A += np.tril(0.018 * (col_m + 1) / (row_m + 2) - 0.004j * (row_m + col_m), k=-1)
    B = (0.08 * (row_mn + 1) / (col_mn + 2) + 0.03j * (col_mn + 1)).astype(np.complex128)
    C = (0.06 * (col_nm + 1) / (row_nm + 2) - 0.025j * (row_nm + 1)).astype(np.complex128)
    D = (np.eye(n) * (1.1 + 0.05j) + 0.015 * (row_n + col_n + 1)).astype(np.complex128)

    A = A.astype(complex_dtype)
    B = B.astype(complex_dtype)
    C = C.astype(complex_dtype)
    D = D.astype(complex_dtype)
    matrix = sparse.bmat(
        (
            (sparse.csc_matrix(A), sparse.csc_matrix(B)),
            (sparse.csc_matrix(C), sparse.csc_matrix(D)),
        ),
        format="csc",
        dtype=complex_dtype,
    )
    schur_indices = np.arange(m, m + n, dtype=int)

    mumps = Mumps(matrix, sym=0, par=1)
    set_icntl(mumps, 4, 0)
    set_icntl(mumps, 3, 0)
    set_schur_centralized_by_column(mumps, schur_indices)
    set_job(mumps, 1)
    set_icntl(mumps, 7, 5)
    mumps_schur_complement_inplace(mumps, schur_indices)
    schur = get_schur_complement(mumps)

    expected = D.astype(np.complex128) - C.astype(np.complex128) @ np.linalg.solve(
        A.astype(np.complex128),
        B.astype(np.complex128),
    )
    relative_error = float(np.linalg.norm(schur - expected) / max(1.0, np.linalg.norm(expected)))
    relative_tolerance = float(2.0 * np.sqrt(np.finfo(real_dtype).eps))

    return MumpsSchurComplementDemoResult(
        matrix=matrix,
        schur_indices=schur_indices,
        schur=schur,
        expected_schur=expected,
        relative_error=relative_error,
        relative_tolerance=relative_tolerance,
        dtype=complex_dtype,
    )


def gaussian_beam_source_profiles(
    *,
    ny: int,
    y_coordinates: Sequence[float],
    y_focus: Sequence[float],
    z_source: float,
    z_focus: float,
    wavelength: float,
    dx: float,
    epsilon_bg: complex,
    numerical_aperture: float,
    yBC: str = "PEC",
) -> GaussianBeamProfiles:
    """Build the Gaussian-beam line sources from the Julia reflection example.

    This ports the source/projection construction from
    ``examples/2d_reflection_matrix_Gaussian_beams``.  The returned
    ``B_low`` is meant to be placed on one z-plane as a ``Source_struct``; the
    matching output projection is expected to satisfy ``C = transpose(B)`` up
    to roundoff for reciprocal systems.
    """

    if ny <= 0:
        raise ValueError("ny must be positive.")
    if dx <= 0:
        raise ValueError("dx must be positive.")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    if numerical_aperture <= 0:
        raise ValueError("numerical_aperture must be positive.")

    y = np.asarray(y_coordinates, dtype=float).reshape(-1, 1)
    if y.shape[0] != ny:
        raise ValueError("y_coordinates length must equal ny.")
    y_focus_arr = np.asarray(y_focus, dtype=float).reshape(1, -1)
    if y_focus_arr.shape[1] == 0:
        raise ValueError("y_focus must contain at least one focal position.")

    beam_radius = wavelength / (np.pi * numerical_aperture)
    E_yf = np.exp(-((y - y_focus_arr) ** 2) / (beam_radius**2))

    k0dx = (2 * np.pi / wavelength) * dx
    channels = mesti_build_channels(ny, yBC, k0dx, epsilon_bg)
    if channels.N_prop is None or channels.N_prop == 0:
        raise ValueError("The Gaussian example requires at least one propagating channel.")

    f_transverse = channels.f_x_m(channels.kydx_prop)
    sqrt_nu = np.asarray(channels.sqrt_nu_prop, dtype=np.complex128).reshape(-1, 1)
    v_f = (sqrt_nu * f_transverse.conj().T) @ E_yf

    kz = np.asarray(channels.kzdx_prop, dtype=np.complex128).reshape(-1, 1) / dx
    v_s = np.exp(1j * kz * (z_source - z_focus)) * v_f
    B_low = (f_transverse * sqrt_nu.T) @ v_s

    psi_yf = E_yf
    v_f_tilde = (sqrt_nu * f_transverse.conj().T) @ psi_yf
    v_d = np.exp(-1j * kz * (z_source - z_focus)) * v_f_tilde
    C_low = v_d.conj().T @ (sqrt_nu * f_transverse.conj().T)
    mismatch = float(np.max(np.abs(C_low - B_low.T)))

    return GaussianBeamProfiles(
        B_low=np.asarray(B_low, dtype=np.complex128),
        C_low=np.asarray(C_low, dtype=np.complex128),
        y_focus=y_focus_arr.reshape(-1),
        beam_radius=float(beam_radius),
        channels=channels,
        transpose_mismatch_max=mismatch,
    )


def reflection_matrix_gaussian_beams(
    *,
    epsilon_xx: Any,
    wavelength: float,
    dx: float,
    pml_npixels: int,
    y_focus: Sequence[float],
    z_focus: float,
    source_plane_index: int,
    epsilon_bg: complex = 1.0,
    numerical_aperture: float = 0.5,
    y_coordinates: Sequence[float] | None = None,
    z_coordinates: Sequence[float] | None = None,
    yBC: str = "PEC",
    zBC: str = "PEC",
    solver: str | None = "scipy",
    compute_field_profile: bool = True,
) -> GaussianReflectionResult:
    """Compute the reduced 2D Gaussian-beam reflection-matrix example.

    ``source_plane_index`` is zero-based, following the rest of the Python
    port.  The Julia example uses a circular ``Ball`` only to generate
    ``epsilon_xx``; this helper accepts the permittivity profile directly so
    fixtures generated from Julia subpixel smoothing can be replayed in Python.
    """

    epsilon = np.asarray(epsilon_xx, dtype=np.complex128)
    if epsilon.ndim != 2:
        raise ValueError("epsilon_xx must be a 2D TM permittivity array.")
    if pml_npixels < 0:
        raise ValueError("pml_npixels must be nonnegative.")
    ny, nz = epsilon.shape
    if source_plane_index < 0 or source_plane_index >= nz:
        raise ValueError("source_plane_index must select a z plane in epsilon_xx.")

    y = (
        np.asarray(y_coordinates, dtype=float).reshape(-1)
        if y_coordinates is not None
        else dx * np.arange(1, ny + 1, dtype=float)
    )
    z = (
        np.asarray(z_coordinates, dtype=float).reshape(-1)
        if z_coordinates is not None
        else dx * np.arange(1, nz + 1, dtype=float)
    )
    if y.size != ny:
        raise ValueError("y_coordinates length must match epsilon_xx.shape[0].")
    if z.size != nz:
        raise ValueError("z_coordinates length must match epsilon_xx.shape[1].")

    profiles = gaussian_beam_source_profiles(
        ny=ny,
        y_coordinates=y,
        y_focus=y_focus,
        z_source=float(z[source_plane_index]),
        z_focus=float(z_focus),
        wavelength=float(wavelength),
        dx=float(dx),
        epsilon_bg=epsilon_bg,
        numerical_aperture=float(numerical_aperture),
        yBC=yBC,
    )
    source = Source_struct(
        pos=[np.array([0, source_plane_index, ny - 1, source_plane_index], dtype=int)],
        data=[profiles.B_low],
    )

    pml = PML(int(pml_npixels), direction="all")
    base_nz = source_plane_index + 1 + int(pml_npixels)
    base_syst = Syst(
        epsilon_xx=np.full((ny, base_nz), epsilon_bg, dtype=np.complex128),
        wavelength=float(wavelength),
        dx=float(dx),
        yBC=yBC,
        zBC=zBC,
        PML=pml,
    )
    reference, reference_info = mesti(
        base_syst,
        source,
        C="transpose(B)",
        opts=Opts(solver=solver, verbal=False, prefactor=-2j),
    )

    syst = Syst(
        epsilon_xx=epsilon,
        wavelength=float(wavelength),
        dx=float(dx),
        yBC=yBC,
        zBC=zBC,
        PML=PML(int(pml_npixels), direction="all"),
    )
    reflection, reflection_info = mesti(
        syst,
        source,
        C="transpose(B)",
        D=reference,
        opts=Opts(solver=solver, verbal=False, prefactor=-2j),
    )

    field_profiles = None
    field_info = None
    if compute_field_profile:
        field_profiles, field_info = mesti(
            syst,
            source,
            opts=Opts(
                solver=solver,
                verbal=False,
                prefactor=-2j,
                exclude_PML_in_field_profiles=True,
            ),
        )

    return GaussianReflectionResult(
        source=source,
        profiles=profiles,
        reference=reference,
        reflection=reflection,
        field_profiles=field_profiles,
        reference_info=reference_info,
        reflection_info=reflection_info,
        field_info=field_info,
    )
