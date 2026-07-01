"""Transverse and longitudinal channel setup for the Python MESTI port."""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np

from .boundary import convert_BC, convert_BC_to_transverse
from .types import Channels_one_sided, Channels_two_sided, Side, Syst


def _is_number(value: object) -> bool:
    return isinstance(value, numbers.Number)


def convert_BC_1d(BC: str | numbers.Number, direction: str) -> str | numbers.Number:
    """Normalize 1D transverse boundary labels."""

    if _is_number(BC):
        return BC
    bc = str(BC).lower()
    if bc == "dirichlet":
        return "Dirichlet"
    if bc == "neumann":
        return "Neumann"
    if bc == "dirichletneumann":
        return "DirichletNeumann"
    if bc == "neumanndirichlet":
        return "NeumannDirichlet"
    if bc == "periodic":
        return "periodic"
    if bc == "bloch":
        raise ValueError(
            "To use Bloch periodic boundary condition in "
            f"{direction}-direction, set {direction}BC to k{direction}_B*p_{direction}."
        )
    raise ValueError(f'Input argument {direction}BC = "{BC}" is not a supported option.')


def _as_column_modes(kxdx: Any) -> np.ndarray:
    values = np.asarray(kxdx, dtype=np.complex128)
    if values.ndim == 0:
        values = values.reshape(1)
    return values.reshape(1, -1)


def mesti_build_transverse_function(
    nx: int,
    xBC: str | numbers.Number,
    n0: float = 0,
    offset: bool = False,
) -> tuple[Any, np.ndarray]:
    """Set up 1D transverse mode functions and wave numbers."""

    if nx < 0:
        raise ValueError("Input argument nx must be a natural number.")

    xBC = convert_BC_1d(xBC, "x")
    if _is_number(xBC):
        ka_x = xBC
        xBC = "Bloch"
    elif xBC == "periodic":
        ka_x = 0
        xBC = "Bloch"
    else:
        ka_x = 0

    if xBC == "Bloch":
        ind_zero_kx = round((nx + 1) / 2) if nx % 2 == 1 else round(nx / 2)
        kxdx_all = (ka_x / nx) + (np.arange(1, nx + 1) - ind_zero_kx) * (2 * np.pi / nx)
    elif xBC == "Dirichlet":
        kxdx_all = np.arange(0, nx + 1) * (np.pi / (nx + 1))
    elif xBC == "Neumann":
        kxdx_all = np.arange(0, nx) * (np.pi / nx)
    elif xBC == "DirichletNeumann":
        kxdx_all = (np.arange(0, nx) + 0.5) * (np.pi / (nx + 0.5))
    elif xBC == "NeumannDirichlet":
        kxdx_all = (np.arange(0, nx) + 0.5) * (np.pi / (nx + 0.5))
    else:
        raise ValueError(f"Input argument xBC = {xBC} is not a supported option.")

    n = np.arange(1, nx + 1, dtype=float).reshape(-1, 1)

    def fun_f_1d(kxdx: Any) -> np.ndarray:
        k = _as_column_modes(kxdx)
        if xBC == "Bloch":
            return np.exp((n + 0.5 * bool(offset) - n0) * (1j * k)) / np.sqrt(nx)
        if xBC == "Dirichlet":
            return 1j * np.sin(n * k) * np.sqrt(2 / (nx + 1))
        if xBC == "Neumann":
            n_half = (np.arange(0.5, nx + 0.5, dtype=float)).reshape(-1, 1)
            zero_correction = (k == 0) * (1 - np.sqrt(1 / 2))
            return (np.cos(n_half * k) - zero_correction) * np.sqrt(2 / nx)
        if xBC == "DirichletNeumann":
            return 1j * np.sin(n * k) * np.sqrt(2 / (nx + 0.5))
        if xBC == "NeumannDirichlet":
            n_half = (np.arange(0.5, nx + 0.5, dtype=float)).reshape(-1, 1)
            return np.cos(n_half * k) * np.sqrt(2 / (nx + 0.5))
        raise ValueError(f"Input argument xBC = {xBC} is not a supported option.")

    return fun_f_1d, np.asarray(kxdx_all, dtype=float)


def mesti_build_transverse_function_derivative(
    nx: int,
    xBC: str | numbers.Number,
    n0: float = 0,
    changegrid: float = 0,
) -> Any:
    """Set up finite-difference derivatives of 1D transverse functions."""

    if nx < 0:
        raise ValueError("Input argument nx must be a natural number.")

    xBC = convert_BC_1d(xBC, "x")
    if _is_number(xBC) or xBC == "periodic":
        xBC = "Bloch"

    def fun_df_1d(kxdx: Any) -> np.ndarray:
        k = _as_column_modes(kxdx)
        if xBC == "Bloch":
            n2 = np.arange(2, nx + 2, dtype=float).reshape(-1, 1)
            n1 = np.arange(1, nx + 1, dtype=float).reshape(-1, 1)
            return (
                np.exp((n2 - n0) * (1j * k)) / np.sqrt(nx)
                - np.exp((n1 - n0) * (1j * k)) / np.sqrt(nx)
            )
        if xBC == "Dirichlet":
            n2 = np.arange(2 - changegrid, nx + 2, dtype=float).reshape(-1, 1)
            n1 = np.arange(1 - changegrid, nx + 1, dtype=float).reshape(-1, 1)
            return 1j * (np.sin(n2 * k) - np.sin(n1 * k)) * np.sqrt(2 / (nx + 1))
        if xBC == "Neumann":
            n2 = np.arange(1.5, nx + 1.5 - changegrid, dtype=float).reshape(-1, 1)
            n1 = np.arange(0.5, nx + 0.5 - changegrid, dtype=float).reshape(-1, 1)
            zero_correction = (k == 0) * (1 - np.sqrt(1 / 2))
            return (
                (np.cos(n2 * k) - zero_correction)
                - (np.cos(n1 * k) - zero_correction)
            ) * np.sqrt(2 / nx)
        if xBC == "DirichletNeumann":
            n2 = np.arange(2 - changegrid, nx + 2 - changegrid, dtype=float).reshape(-1, 1)
            n1 = np.arange(1 - changegrid, nx + 1 - changegrid, dtype=float).reshape(-1, 1)
            return 1j * (np.sin(n2 * k) - np.sin(n1 * k)) * np.sqrt(2 / (nx + 0.5))
        if xBC == "NeumannDirichlet":
            n2 = np.arange(1.5, nx + 1.5, dtype=float).reshape(-1, 1)
            n1 = np.arange(0.5, nx + 0.5, dtype=float).reshape(-1, 1)
            return (np.cos(n2 * k) - np.cos(n1 * k)) * np.sqrt(2 / (nx + 0.5))
        raise ValueError(f"Input argument xBC = {xBC} is not a supported option.")

    return fun_df_1d


def _is_real_scalar(value: Any) -> bool:
    return np.isscalar(value) and np.isrealobj(value)


def _check_bc_and_grid_2d(
    BC: str | numbers.Number,
    n_Ex: int,
    n_Ey: int,
    direction: str,
) -> None:
    """Validate the two staggered transverse grids used by 3D channels."""

    if _is_number(BC):
        if n_Ex != n_Ey:
            raise ValueError(
                f"Number of grids along {direction} from epsilon_xx and epsilon_yy "
                "must match for Bloch periodic boundary conditions."
            )
        return

    if BC in {"periodic", "PECPMC", "PMCPEC"}:
        if n_Ex != n_Ey:
            raise ValueError(
                f"Number of grids along {direction} from epsilon_xx and epsilon_yy "
                f"must match for {BC} boundary conditions."
            )
        return

    if direction == "x":
        primary, secondary, name = n_Ex, n_Ey, "epsilon_xx"
    elif direction == "y":
        primary, secondary, name = n_Ey, n_Ex, "epsilon_yy"
    else:
        raise ValueError(f"Unsupported direction {direction!r}.")

    if BC == "PEC" and primary != secondary + 1:
        raise ValueError(
            f"Number of grids along {direction} from {name} must be one larger "
            f"than the other electric component for PEC."
        )
    if BC == "PMC" and primary != secondary - 1:
        raise ValueError(
            f"Number of grids along {direction} from {name} must be one smaller "
            f"than the other electric component for PMC."
        )


def mesti_setup_longitudinal(
    k0dx: float | complex,
    epsilon_bg: float | complex,
    kxdx_all: np.ndarray | None,
    kydx_all: np.ndarray,
    kLambda_x: float | complex | None = None,
    kLambda_y: float | complex | None = None,
    ind_zero_kx: int | None = None,
    ind_zero_ky: int | None = None,
    use_continuous_dispersion: bool = False,
) -> Side:
    """Set up longitudinal channel metadata for one homogeneous side."""

    use_2D_TM = kxdx_all is None and kLambda_x is None and ind_zero_kx is None
    if not use_2D_TM:
        kxdx_all = np.asarray(kxdx_all, dtype=float).reshape(-1)
        nx = kxdx_all.size
        if nx <= 0:
            raise ValueError("kxdx_all must contain at least one transverse x channel.")
    kydx_all = np.asarray(kydx_all, dtype=float)
    ny = kydx_all.size
    k0dx2_epsilon = (k0dx**2) * epsilon_bg
    side = Side()

    if not use_continuous_dispersion:
        if use_2D_TM:
            sin_kzdx_over_two_sq = 0.25 * k0dx2_epsilon - np.sin(kydx_all / 2) ** 2
        else:
            sin_kx = np.sin(kxdx_all[:, np.newaxis] / 2) ** 2
            sin_ky = np.sin(kydx_all[np.newaxis, :] / 2) ** 2
            # Julia flattens the transverse (kx, ky) grid column-major, so kx
            # varies fastest for channel index zero, then ky increments.
            sin_kzdx_over_two_sq = (
                0.25 * k0dx2_epsilon - sin_kx - sin_ky
            ).reshape(nx * ny, order="F")

        side.kzdx_all = 2 * np.arcsin(np.sqrt(np.asarray(sin_kzdx_over_two_sq, dtype=np.complex128)))
        side.ind_prop = np.flatnonzero(
            (np.real(sin_kzdx_over_two_sq) > 0) & (np.real(sin_kzdx_over_two_sq) < 1)
        )

        real_threshold = 4 if use_2D_TM else 6
        needs_flip = (not _is_real_scalar(k0dx2_epsilon)) or (
            _is_real_scalar(k0dx2_epsilon) and np.real(k0dx2_epsilon) > real_threshold
        )
        if needs_flip:
            z = np.asarray(sin_kzdx_over_two_sq)
            ind_flip = np.flatnonzero(
                ((np.real(z) < 0) & (np.imag(z) < 0))
                | ((np.real(z) > 1) & (np.imag(z) <= 0))
            )
            side.kzdx_all[ind_flip] = -side.kzdx_all[ind_flip]
    else:
        if use_2D_TM:
            kzdx2 = k0dx2_epsilon - kydx_all**2
        else:
            # Continuous-dispersion channels use the same x-fastest flattening
            # as the finite-difference branch so metadata indices agree.
            kzdx2 = (
                k0dx2_epsilon - kxdx_all[:, np.newaxis] ** 2 - kydx_all[np.newaxis, :] ** 2
            ).reshape(nx * ny, order="F")
        side.kzdx_all = np.sqrt(np.asarray(kzdx2, dtype=np.complex128))
        side.ind_prop = np.flatnonzero(np.real(kzdx2) > 0)
        if not _is_real_scalar(k0dx2_epsilon):
            ind_flip = np.flatnonzero((np.real(kzdx2) < 0) & (np.imag(kzdx2) < 0))
            side.kzdx_all[ind_flip] = -side.kzdx_all[ind_flip]

    side.N_prop = int(len(side.ind_prop))
    side.kzdx_prop = side.kzdx_all[side.ind_prop]
    if use_2D_TM:
        side.kxdx_prop = None
        side.kydx_prop = kydx_all[side.ind_prop]
    else:
        # Propagating channel indices are zero-based Python equivalents of
        # Julia's x-fastest linear index into the (kx, ky) channel grid.
        side.kxdx_prop = kxdx_all[side.ind_prop % nx]
        side.kydx_prop = kydx_all[side.ind_prop // nx]

    side.sqrt_nu_prop = np.sqrt(np.sin(side.kzdx_prop))

    if not use_2D_TM:
        side.ind_prop_conj = np.arange(side.N_prop)
    elif kLambda_y is None:
        side.ind_prop_conj = np.arange(side.N_prop)
    elif kLambda_y == 0:
        if (ind_zero_ky in set(side.ind_prop.tolist())) or (side.N_prop % 2 == 0):
            side.ind_prop_conj = np.arange(side.N_prop - 1, -1, -1)
        else:
            side.ind_prop_conj = np.concatenate(
                [np.arange(side.N_prop - 2, -1, -1), np.array([side.N_prop - 1])]
            )
    else:
        side.ind_prop_conj = np.arange(side.N_prop)

    return side


def _copy_side_to_one_sided(channels: Channels_one_sided, side: Side) -> Channels_one_sided:
    for name in (
        "N_prop",
        "kzdx_all",
        "ind_prop",
        "kxdx_prop",
        "kydx_prop",
        "kzdx_prop",
        "sqrt_nu_prop",
        "ind_prop_conj",
    ):
        setattr(channels, name, getattr(side, name))
    return channels


def _mesti_build_channels_full(
    nx_Ex: int | None,
    nx_Ey: int | None,
    xBC: str | numbers.Number | None,
    ny_Ex: int,
    ny_Ey: int | None,
    yBC: str | numbers.Number,
    k0dx: float | complex,
    epsilon_low: float | complex,
    epsilon_high: float | complex | None = None,
    use_continuous_dispersion: bool = False,
    n0: float | None = 0,
    m0: float = 0,
) -> Channels_one_sided | Channels_two_sided:
    use_2D_TM = nx_Ex is None and nx_Ey is None and ny_Ex is not None and ny_Ey is None
    if use_2D_TM:
        if ny_Ex <= 0:
            raise ValueError("Input argument ny_Ex must be a positive integer scalar.")
    else:
        if nx_Ex is None or nx_Ey is None or xBC is None or ny_Ey is None:
            raise ValueError("3D channel setup requires nx_Ex, nx_Ey, xBC, ny_Ex, ny_Ey, and yBC.")
        if nx_Ex <= 0 or nx_Ey <= 0 or ny_Ex <= 0 or ny_Ey <= 0:
            raise ValueError("3D channel grid counts must be positive integer scalars.")

    two_sided = epsilon_high is not None
    if not use_2D_TM:
        xBC = convert_BC(xBC, "x")
        _check_bc_and_grid_2d(xBC, nx_Ex, nx_Ey, "x")
        BC_x_x = convert_BC_to_transverse(xBC, "x", "x")
        BC_y_x = convert_BC_to_transverse(xBC, "y", "x")
        BC_z_x = convert_BC_to_transverse(xBC, "z", "x")
    yBC = convert_BC(yBC, "y")
    if not use_2D_TM:
        _check_bc_and_grid_2d(yBC, ny_Ex, ny_Ey, "y")
        BC_y_y = convert_BC_to_transverse(yBC, "y", "y")
        BC_z_y = convert_BC_to_transverse(yBC, "z", "y")
    BC_x_y = convert_BC_to_transverse(yBC, "x", "y")

    kLambda_x = None
    kLambda_y = None
    ind_zero_kx = None
    ind_zero_ky = None
    if not use_2D_TM:
        if _is_number(xBC):
            # A numeric xBC is already the dimensionless Bloch phase
            # kx_B*periodicity.  The same phase is used by both staggered x grids.
            kLambda_x = xBC
            xBC_for_zero = "Bloch"
        elif str(xBC).lower() == "bloch":
            raise ValueError(
                "To use Bloch periodic boundary condition in mesti_build_channels(), "
                "set xBC to kx_B*Lambda_x."
            )
        elif xBC == "periodic":
            kLambda_x = 0
            xBC_for_zero = "Bloch"
        else:
            xBC_for_zero = xBC

    if _is_number(yBC):
        # Julia passes Bloch boundaries into channel setup as the dimensionless
        # phase ky_B * Lambda_y; the public Syst.ky_B path is converted before
        # reaching this helper.
        kLambda_y = yBC
        yBC_for_zero = "Bloch"
    elif str(yBC).lower() == "bloch":
        raise ValueError(
            "To use Bloch periodic boundary condition in mesti_build_channels(), "
            "set yBC to ky_B*Lambda_y."
        )
    elif yBC == "periodic":
        kLambda_y = 0
        yBC_for_zero = "Bloch"
    else:
        yBC_for_zero = yBC

    channels: Channels_one_sided | Channels_two_sided
    channels = Channels_two_sided() if two_sided else Channels_one_sided()
    if use_2D_TM:
        # For 2D TM Ex(y,z), Julia applies opts.m0 as a transverse-mode origin
        # shift in f_x_m even though no x grid is present in the reduced problem.
        channels.f_x_m, channels.kydx_all = mesti_build_transverse_function(ny_Ex, BC_x_y, m0)
        channels.kxdx_all = None
    else:
        # 3D channels follow the Yee staggering in Julia: Ex uses offset x
        # modes, Ey uses offset y modes, and Ez reuses the Ey/Ex transverse
        # functions on the matching staggered grids.
        channels.f_x_n, channels.kxdx_all = mesti_build_transverse_function(nx_Ex, BC_x_x, n0 or 0, True)
        channels.f_x_m, _ = mesti_build_transverse_function(ny_Ex, BC_x_y, n0 or 0)
        channels.f_y_n, _ = mesti_build_transverse_function(nx_Ey, BC_y_x, n0 or 0)
        channels.f_y_m, channels.kydx_all = mesti_build_transverse_function(ny_Ey, BC_y_y, m0, True)
        channels.f_z_n = channels.f_y_n
        channels.f_z_m = channels.f_x_m
        channels.df_z_n = mesti_build_transverse_function_derivative(nx_Ey, BC_z_x, n0 or 0, 1)
        channels.df_z_m = mesti_build_transverse_function_derivative(ny_Ex, BC_z_y, m0, 1)

    if (not use_2D_TM) and xBC_for_zero == "Bloch":
        ind_zero_kx = int(round((nx_Ex + 1) / 2) if nx_Ex % 2 == 1 else round(nx_Ex / 2)) - 1

    if yBC_for_zero == "Bloch":
        ind_zero_ky = int(round((ny_Ex + 1) / 2) if ny_Ex % 2 == 1 else round(ny_Ex / 2)) - 1

    side = mesti_setup_longitudinal(
        k0dx,
        epsilon_low,
        channels.kxdx_all,
        channels.kydx_all,
        kLambda_x,
        kLambda_y,
        ind_zero_kx,
        ind_zero_ky,
        use_continuous_dispersion,
    )

    if two_sided:
        channels.low = side
        if epsilon_high == epsilon_low:
            channels.high = side
        else:
            channels.high = mesti_setup_longitudinal(
                k0dx,
                epsilon_high,
                channels.kxdx_all,
                channels.kydx_all,
                kLambda_x,
                kLambda_y,
                ind_zero_kx,
                ind_zero_ky,
                use_continuous_dispersion,
            )
    else:
        _copy_side_to_one_sided(channels, side)
    return channels


def _mesti_build_channels_from_syst(syst: Syst) -> Channels_one_sided | Channels_two_sided:
    epsilon_xx = np.asarray(syst.epsilon_xx)
    if syst.epsilon_low is None:
        raise ValueError('Input argument syst must have field "epsilon_low".')
    if syst.wavelength is None:
        raise ValueError('Input argument syst must have field "wavelength".')
    if syst.dx is None or syst.dx <= 0:
        raise ValueError("syst.dx must be a positive scalar.")

    k0dx = (2 * np.pi / syst.wavelength) * syst.dx
    if epsilon_xx.ndim == 2:
        ny_Ex = epsilon_xx.shape[0]
        if syst.ky_B is not None:
            yBC = syst.ky_B * (ny_Ex * syst.dx)
        elif syst.yBC is not None:
            yBC = syst.yBC
        else:
            raise ValueError('Input argument syst must have non-empty field "yBC".')

        return _mesti_build_channels_full(
            None,
            None,
            None,
            ny_Ex,
            None,
            yBC,
            k0dx,
            syst.epsilon_low,
            syst.epsilon_high,
            False,
            None,
            0,
        )

    if epsilon_xx.ndim != 3:
        raise NotImplementedError("Only 2D TM and 3D diagonal channel setup are ported so far.")
    if syst.epsilon_yy is None:
        raise ValueError("3D channel setup requires syst.epsilon_yy.")
    epsilon_yy = np.asarray(syst.epsilon_yy)
    if epsilon_yy.ndim != 3:
        raise ValueError("3D channel setup requires a 3D syst.epsilon_yy array.")
    nx_Ex, ny_Ex, _ = epsilon_xx.shape
    nx_Ey, ny_Ey, _ = epsilon_yy.shape
    if syst.kx_B is not None:
        xBC = syst.kx_B * (nx_Ex * syst.dx)
    elif syst.xBC is not None:
        xBC = syst.xBC
    else:
        raise ValueError('Input argument syst must have non-empty field "xBC".')
    if syst.ky_B is not None:
        yBC = syst.ky_B * (ny_Ey * syst.dx)
    elif syst.yBC is not None:
        yBC = syst.yBC
    else:
        raise ValueError('Input argument syst must have non-empty field "yBC".')

    return _mesti_build_channels_full(
        nx_Ex,
        nx_Ey,
        xBC,
        ny_Ex,
        ny_Ey,
        yBC,
        k0dx,
        syst.epsilon_low,
        syst.epsilon_high,
        False,
        0,
        0,
    )


def mesti_build_channels(*args: Any, **kwargs: Any) -> Channels_one_sided | Channels_two_sided:
    """Build channel metadata for supported 2D TM and diagonal 3D systems."""

    if len(args) == 1 and isinstance(args[0], Syst):
        return _mesti_build_channels_from_syst(args[0])

    full_3d_call = len(args) >= 8 or any(
        name in kwargs for name in ("nx_Ey", "xBC", "ny_Ey")
    )
    if args and isinstance(args[0], int) and not full_3d_call:
        ny_Ex = args[0]
        yBC = args[1] if len(args) > 1 else kwargs.pop("yBC")
        k0dx = args[2] if len(args) > 2 else kwargs.pop("k0dx")
        epsilon_low = args[3] if len(args) > 3 else kwargs.pop("epsilon_low")
        epsilon_high = args[4] if len(args) > 4 else kwargs.pop("epsilon_high", None)
        use_continuous_dispersion = (
            args[5] if len(args) > 5 else kwargs.pop("use_continuous_dispersion", False)
        )
        m0 = args[6] if len(args) > 6 else kwargs.pop("m0", 0)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        return _mesti_build_channels_full(
            None,
            None,
            None,
            ny_Ex,
            None,
            yBC,
            k0dx,
            epsilon_low,
            epsilon_high,
            use_continuous_dispersion,
            None,
            m0,
        )

    return _mesti_build_channels_full(*args, **kwargs)
