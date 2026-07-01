"""Data structures mirrored from the Julia MESTI implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class PML:
    """Perfectly matched layer parameters.

    Mirrors ``mutable struct PML`` in ``mesti_build_fdfd_matrix.jl``.  Julia
    leaves fields undefined until parser code fills them, so Python defaults to
    ``None`` rather than guessing behavior.
    """

    npixels: int | None = None
    sigma_max_over_omega: float | None = None
    power_sigma: float | None = None
    alpha_max_over_omega: float | None = None
    power_alpha: float | None = None
    kappa_max: float | None = None
    power_kappa: float | None = None
    direction: str | None = None
    side: str | None = None
    npixels_spacer: int | None = None


@dataclass
class Source_struct:
    """Source/projection sparse data descriptor from ``mesti_main.jl``."""

    pos: list[Any] | None = None
    data: list[Any] | None = None
    ind: list[Any] | None = None
    isempty: bool | None = None


@dataclass
class Syst:
    """System descriptor from ``mesti_main.jl``."""

    epsilon_xx: Any = None
    epsilon_xy: Any = None
    epsilon_xz: Any = None
    epsilon_yx: Any = None
    epsilon_yy: Any = None
    epsilon_yz: Any = None
    epsilon_zx: Any = None
    epsilon_zy: Any = None
    epsilon_zz: Any = None
    length_unit: str | None = None
    wavelength: Any = None
    dx: float | None = None
    xBC: str | None = None
    yBC: str | None = None
    kx_B: Any = None
    ky_B: Any = None
    zBC: str | None = None
    kz_B: Any = None
    PML: PML | list[PML] | None = None
    PML_type: str | None = None
    epsilon_low: Any = None
    epsilon_high: Any = None
    zPML: PML | list[PML] | None = None


@dataclass
class Matrices:
    """Container for matrices used by ``mesti_matrix_solver``."""

    A: Any = None
    B: Any = None
    C: Any = None
    D: Any = None


@dataclass
class Opts:
    """Computation options from ``mesti_matrix_solver.jl``."""

    is_symmetric_A: int | None = None
    verbal: int | bool | None = None
    prefactor: Any = None
    solver: str | None = None
    method: str | None = None
    clear_BC: int | bool | None = None
    clear_syst: int | bool | None = None
    clear_memory: int | bool | None = None
    verbal_solver: int | bool | None = None
    # Explicit single precision is supported only for the mumpspy backend.
    use_single_precision_MUMPS: int | bool | None = None
    use_METIS: int | bool | None = None
    # Explicit RHS batch-width override.  None uses solver.py's dense/sparse defaults.
    nrhs: int | None = None
    # cuDSS-specific speed/memory controls.  They are accepted only with
    # solver="cudss" and are applied inside cudss_backend.py.
    cudss_use_single_precision: int | bool | None = None
    cudss_use_hybrid_memory: int | bool | None = None
    cudss_hybrid_device_memory_limit: int | str | None = None
    cudss_register_cuda_memory: int | bool | None = None
    store_ordering: int | bool | None = None
    ordering: Any = None
    analysis_only: int | bool | None = None
    nthreads_OMP: int | None = None
    iterative_refinement: int | bool | None = None
    use_L0_threads: int | bool | None = None
    write_LU_factor_to_disk: int | bool | None = None
    exclude_PML_in_field_profiles: int | bool | None = None
    return_field_profile: int | bool | None = None
    use_given_ordering: int | bool | None = None
    n0: float | None = None
    m0: float | None = None
    use_continuous_dispersion: int | bool | None = None
    symmetrize_K: int | bool | None = None
    nz_low: int | None = None
    nz_high: int | None = None
    use_BLR: int | bool | None = None
    threshold_BLR: float | None = None
    icntl_36: int | None = None
    icntl_38: int | None = None


@dataclass
class Info:
    """Information returned by MESTI solver routines."""

    opts: Opts | None = None
    timing_total: float | None = None
    timing_init: float | None = None
    timing_build: float | None = None
    timing_analyze: float | None = None
    timing_factorize: float | None = None
    timing_solve: float | None = None
    ordering_method: Any = None
    ordering: Any = None
    itr_ref_nsteps: int | None = None
    itr_ref_omega_1: Any = None
    itr_ref_omega_2: Any = None
    xPML: list[PML] | None = None
    yPML: list[PML] | None = None
    zPML: list[PML] | None = None
    ind_in_trivial_ch: list[int] | None = None
    ind_out_trivial_ch: list[int] | None = None
    ind_in_nontrivial_ch: list[int] | None = None
    ind_out_nontrivial_ch: list[int] | None = None


@dataclass
class channel_type:
    """Channel selection by side and polarization from ``mesti2s.jl``."""

    side: str | None = None
    polarization: str | None = None


@dataclass
class channel_index:
    """Explicit channel indices from ``mesti2s.jl``."""

    ind_low_s: list[int] | None = None
    ind_low_p: list[int] | None = None
    ind_high_s: list[int] | None = None
    ind_high_p: list[int] | None = None
    ind_low: list[int] | None = None
    ind_high: list[int] | None = None


@dataclass
class wavefront:
    """Linear combinations of propagating channels from ``mesti2s.jl``."""

    v_low_s: Any = None
    v_low_p: Any = None
    v_high_s: Any = None
    v_high_p: Any = None
    v_low: Any = None
    v_high: Any = None


@dataclass
class Side:
    """One homogeneous side's channel metadata."""

    N_prop: int | None = None
    kzdx_all: Any = None
    ind_prop: list[int] | None = None
    kxdx_prop: Any = None
    kydx_prop: Any = None
    kzdx_prop: Any = None
    sqrt_nu_prop: Any = None
    ind_prop_conj: list[int] | None = None


@dataclass
class Channels_two_sided:
    """Channel metadata for two-sided scattering geometry."""

    f_x_n: Callable[..., Any] | None = None
    f_y_n: Callable[..., Any] | None = None
    f_z_n: Callable[..., Any] | None = None
    df_z_n: Callable[..., Any] | None = None
    f_x_m: Callable[..., Any] | None = None
    f_y_m: Callable[..., Any] | None = None
    f_z_m: Callable[..., Any] | None = None
    df_z_m: Callable[..., Any] | None = None
    kxdx_all: Any = None
    kydx_all: Any = None
    low: Side | None = None
    high: Side | None = None


@dataclass
class Channels_one_sided:
    """Channel metadata for one-sided scattering geometry."""

    f_x_n: Callable[..., Any] | None = None
    f_y_n: Callable[..., Any] | None = None
    f_z_n: Callable[..., Any] | None = None
    df_z_n: Callable[..., Any] | None = None
    f_x_m: Callable[..., Any] | None = None
    f_y_m: Callable[..., Any] | None = None
    f_z_m: Callable[..., Any] | None = None
    df_z_m: Callable[..., Any] | None = None
    kxdx_all: Any = None
    kydx_all: Any = None
    N_prop: int | None = None
    kzdx_all: Any = None
    ind_prop: list[int] | None = None
    kxdx_prop: Any = None
    kydx_prop: Any = None
    kzdx_prop: Any = None
    sqrt_nu_prop: Any = None
    ind_prop_conj: list[int] | None = None
