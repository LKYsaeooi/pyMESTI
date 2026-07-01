"""Finite-difference frequency-domain matrix assembly."""

from __future__ import annotations

from copy import deepcopy
import numbers
from typing import Sequence

import numpy as np
from scipy import sparse

from .boundary import build_ave_x_Ex, build_ddx_E, convert_BC, mesti_set_PML_params
from .types import PML


def _is_number(value: object) -> bool:
    return isinstance(value, numbers.Number)


def _pml_pair(pml: Sequence[PML] | None) -> list[PML]:
    if pml is None:
        return [PML(0), PML(0)]
    pml = list(deepcopy(pml))
    if len(pml) != 2:
        raise ValueError("PML input must contain two PML objects.")
    for layer in pml:
        if layer.npixels is None:
            layer.npixels = 0
    return pml


def _diag(values: np.ndarray) -> sparse.csc_matrix:
    return sparse.diags(np.asarray(values, dtype=np.complex128), 0, format="csc")


def _identity(n: int) -> sparse.csc_matrix:
    return sparse.identity(n, format="csc", dtype=np.complex128)


def _any_pml(pml: Sequence[PML]) -> bool:
    return any((layer.npixels or 0) != 0 for layer in pml)


def _bloch_breaks_symmetry(BC: str | numbers.Number, n: int) -> bool:
    if not _is_number(BC) or n <= 1:
        return False
    return not (np.isclose(BC, 0) or np.isclose(BC, np.pi))


def _bloch_breaks_symmetry_3d(BC: str | numbers.Number, dims: tuple[int, int, int]) -> bool:
    if not _is_number(BC) or not any(dim > 1 for dim in dims):
        return False
    return not (np.isclose(BC, 0) or np.isclose(BC, np.pi))


def _check_bc_and_grid(
    BC: str | numbers.Number,
    n_Ex: int,
    n_Ey: int,
    n_Ez: int,
    direction: str,
) -> None:
    """Validate the Yee-grid staggering required by Julia's 3D builder."""

    if _is_number(BC):
        if n_Ex != n_Ey or n_Ex != n_Ez:
            raise ValueError(
                f"Number of grids along {direction} from epsilon_xx, epsilon_yy, "
                f"and epsilon_zz must match for Bloch periodic boundary conditions."
            )
        return

    if BC in {"periodic", "PECPMC", "PMCPEC"}:
        if n_Ex != n_Ey or n_Ex != n_Ez:
            raise ValueError(
                f"Number of grids along {direction} from epsilon_xx, epsilon_yy, "
                f"and epsilon_zz must match for {BC} boundary conditions."
            )
        return

    if direction == "x":
        primary = n_Ex
        secondary = (n_Ey, n_Ez)
        primary_name = "epsilon_xx"
    elif direction == "y":
        primary = n_Ey
        secondary = (n_Ex, n_Ez)
        primary_name = "epsilon_yy"
    elif direction == "z":
        primary = n_Ez
        secondary = (n_Ex, n_Ey)
        primary_name = "epsilon_zz"
    else:
        raise ValueError(f"Unsupported direction {direction!r}.")

    if BC == "PEC" and any(primary != value + 1 for value in secondary):
        raise ValueError(
            f"Number of grids along {direction} from {primary_name} must be one "
            f"larger than the other electric components for PEC."
        )
    if BC == "PMC" and any(primary != value - 1 for value in secondary):
        raise ValueError(
            f"Number of grids along {direction} from {primary_name} must be one "
            f"smaller than the other electric components for PMC."
        )


def _kron_zyx(z_matrix: sparse.spmatrix, y_matrix: sparse.spmatrix, x_matrix: sparse.spmatrix) -> sparse.csc_matrix:
    """Kronecker expansion for Julia column-major vectors with x fastest."""

    return sparse.kron(z_matrix, sparse.kron(y_matrix, x_matrix, format="csc"), format="csc")


def _zero(rows: int, cols: int) -> sparse.csc_matrix:
    return sparse.csc_matrix((rows, cols), dtype=np.complex128)


def _validate_off_diagonal_3d_component(
    component: np.ndarray | None,
    expected_shape: tuple[int, int, int],
    name: str,
) -> np.ndarray | None:
    if component is None:
        return None
    array = np.asarray(component, dtype=np.complex128)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}; got {array.shape}.")
    return array


def _off_diagonal_material_block(
    left_average: sparse.spmatrix,
    epsilon_component: np.ndarray,
    right_average: sparse.spmatrix,
    k0dx: float | complex,
) -> sparse.csc_matrix:
    return (-(k0dx**2) * (left_average @ _diag(epsilon_component.ravel(order="F")) @ right_average)).tocsc()


def _mesti_build_fdfd_matrix_3d_diagonal(
    epsilon_xx: np.ndarray,
    epsilon_yy: np.ndarray,
    epsilon_zz: np.ndarray,
    k0dx: float | complex,
    xBC: str | numbers.Number,
    yBC: str | numbers.Number,
    zBC: str | numbers.Number,
    xPML: Sequence[PML] | None,
    yPML: Sequence[PML] | None,
    zPML: Sequence[PML] | None,
    use_UPML: bool,
    *,
    epsilon_xy: np.ndarray | None = None,
    epsilon_xz: np.ndarray | None = None,
    epsilon_yx: np.ndarray | None = None,
    epsilon_yz: np.ndarray | None = None,
    epsilon_zx: np.ndarray | None = None,
    epsilon_zy: np.ndarray | None = None,
) -> tuple[sparse.csc_matrix, bool, list[PML], list[PML], list[PML]]:
    """Build Julia-compatible 3D vectorial FDFD assembly."""

    epsilon_xx = np.asarray(epsilon_xx, dtype=np.complex128)
    epsilon_yy = np.asarray(epsilon_yy, dtype=np.complex128)
    epsilon_zz = np.asarray(epsilon_zz, dtype=np.complex128)
    if epsilon_xx.ndim != 3 or epsilon_yy.ndim != 3 or epsilon_zz.ndim != 3:
        raise ValueError("3D vectorial assembly requires three 3D diagonal epsilon arrays.")

    nx_Ex, ny_Ex, nz_Ex = epsilon_xx.shape
    nx_Ey, ny_Ey, nz_Ey = epsilon_yy.shape
    nx_Ez, ny_Ez, nz_Ez = epsilon_zz.shape
    if min(epsilon_xx.shape + epsilon_yy.shape + epsilon_zz.shape) <= 0:
        raise ValueError("3D epsilon arrays must have positive dimensions.")

    xPML = _pml_pair(xPML)
    yPML = _pml_pair(yPML)
    zPML = _pml_pair(zPML)
    xBC_original = xBC
    yBC_original = yBC
    zBC_original = zBC
    xBC = convert_BC(xBC, "x")
    yBC = convert_BC(yBC, "y")
    zBC = convert_BC(zBC, "z")

    if nx_Ey != nx_Ez:
        raise ValueError("Number of x grids from epsilon_yy and epsilon_zz must match.")
    if ny_Ex != ny_Ez:
        raise ValueError("Number of y grids from epsilon_xx and epsilon_zz must match.")
    if nz_Ex != nz_Ey:
        raise ValueError("Number of z grids from epsilon_xx and epsilon_yy must match.")
    _check_bc_and_grid(xBC, nx_Ex, nx_Ey, nx_Ez, "x")
    _check_bc_and_grid(yBC, ny_Ex, ny_Ey, ny_Ez, "y")
    _check_bc_and_grid(zBC, nz_Ex, nz_Ey, nz_Ez, "z")
    epsilon_xy = _validate_off_diagonal_3d_component(epsilon_xy, (nx_Ez, ny_Ez, nz_Ex), "epsilon_xy")
    epsilon_xz = _validate_off_diagonal_3d_component(epsilon_xz, (nx_Ey, ny_Ex, nz_Ey), "epsilon_xz")
    epsilon_yx = _validate_off_diagonal_3d_component(epsilon_yx, (nx_Ez, ny_Ez, nz_Ey), "epsilon_yx")
    epsilon_yz = _validate_off_diagonal_3d_component(epsilon_yz, (nx_Ey, ny_Ex, nz_Ex), "epsilon_yz")
    epsilon_zx = _validate_off_diagonal_3d_component(epsilon_zx, (nx_Ey, ny_Ez, nz_Ey), "epsilon_zx")
    epsilon_zy = _validate_off_diagonal_3d_component(epsilon_zy, (nx_Ez, ny_Ex, nz_Ex), "epsilon_zy")
    include_off_diagonal = any(
        component is not None
        for component in (epsilon_xy, epsilon_xz, epsilon_yx, epsilon_yz, epsilon_zx, epsilon_zy)
    )

    epsilon_bg_x_Ex = [1.0, 1.0]
    epsilon_bg_y_Ey = [1.0, 1.0]
    epsilon_bg_z_Ez = [1.0, 1.0]
    if _any_pml(xPML):
        epsilon_bg_x_Ex = [
            float(np.real(np.mean(epsilon_xx[0, :, :]))),
            float(np.real(np.mean(epsilon_xx[-1, :, :]))),
        ]
        if _is_number(xBC) or xBC == "periodic":
            mean_bg = float(np.mean(epsilon_bg_x_Ex))
            epsilon_bg_x_Ex = [mean_bg, mean_bg]
    if _any_pml(yPML):
        epsilon_bg_y_Ey = [
            float(np.real(np.mean(epsilon_yy[:, 0, :]))),
            float(np.real(np.mean(epsilon_yy[:, -1, :]))),
        ]
        if _is_number(yBC) or yBC == "periodic":
            mean_bg = float(np.mean(epsilon_bg_y_Ey))
            epsilon_bg_y_Ey = [mean_bg, mean_bg]
    if _any_pml(zPML):
        epsilon_bg_z_Ez = [
            float(np.real(np.mean(epsilon_zz[:, :, 0]))),
            float(np.real(np.mean(epsilon_zz[:, :, -1]))),
        ]
        if _is_number(zBC) or zBC == "periodic":
            mean_bg = float(np.mean(epsilon_bg_z_Ez))
            epsilon_bg_z_Ez = [mean_bg, mean_bg]

    xPML = mesti_set_PML_params(xPML, k0dx, epsilon_bg_x_Ex, "x")
    yPML = mesti_set_PML_params(yPML, k0dx, epsilon_bg_y_Ey, "y")
    zPML = mesti_set_PML_params(zPML, k0dx, epsilon_bg_z_Ez, "z")

    ddx_HzEy, _, sx_Ey, sx_Hz, _ = build_ddx_E(nx_Ey, xBC, xPML, "x")
    ddy_HxEz, _, sy_Ez, sy_Hx, _ = build_ddx_E(ny_Ez, yBC, yPML, "y")
    ddz_HyEx, _, sz_Ex, sz_Hy, _ = build_ddx_E(nz_Ex, zBC, zPML, "z")
    if include_off_diagonal:
        avg_x_Ex = build_ave_x_Ex(nx_Ex, xBC, "x")
        avg_y_Ey = build_ave_x_Ex(ny_Ey, yBC, "y")
        avg_z_Ez = build_ave_x_Ex(nz_Ez, zBC, "z")

    ddx_EyHz = (-ddx_HzEy.conjugate().transpose()).tocsc()
    ddy_EzHx = (-ddy_HxEz.conjugate().transpose()).tocsc()
    ddz_ExHy = (-ddz_HyEx.conjugate().transpose()).tocsc()

    nx_Hz = ddx_HzEy.shape[0]
    nx_Hy = nx_Hz
    nx_Hx = nx_Ey
    ny_Hx = ddy_HxEz.shape[0]
    ny_Hz = ny_Hx
    ny_Hy = ny_Ez
    nz_Hy = ddz_HyEx.shape[0]
    nz_Hx = nz_Hy
    nz_Hz = nz_Ex

    nt_Ex = nx_Ex * ny_Ex * nz_Ex
    nt_Ey = nx_Ey * ny_Ey * nz_Ey
    nt_Ez = nx_Ez * ny_Ez * nz_Ez
    nt_Hx = nx_Hx * ny_Hx * nz_Hx
    nt_Hy = nx_Hy * ny_Hy * nz_Hy
    nt_Hz = nx_Hz * ny_Hz * nz_Hz

    # Julia first builds the stretched-coordinate PML operator in 3D.  UPML is
    # a later diagonal row scaling, so both E-side and H-side derivative
    # matrices receive their 1/s factors before curl_H * curl_E is formed.
    ddx_HzEy = _diag(1 / sx_Hz) @ ddx_HzEy
    ddx_HyEz = ddx_HzEy
    ddy_HxEz = _diag(1 / sy_Hx) @ ddy_HxEz
    ddy_HzEx = ddy_HxEz
    ddz_HyEx = _diag(1 / sz_Hy) @ ddz_HyEx
    ddz_HxEy = ddz_HyEx
    ddx_EyHz = _diag(1 / sx_Ey) @ ddx_EyHz
    ddx_EzHy = ddx_EyHz
    ddy_EzHx = _diag(1 / sy_Ez) @ ddy_EzHx
    ddy_ExHz = ddy_EzHx
    ddz_ExHy = _diag(1 / sz_Ex) @ ddz_ExHy
    ddz_EyHx = ddz_ExHy

    Dx_HzEy = _kron_zyx(_identity(nz_Ey), _identity(ny_Ey), ddx_HzEy)
    Dx_HyEz = _kron_zyx(_identity(nz_Ez), _identity(ny_Ez), ddx_HyEz)
    Dy_HxEz = _kron_zyx(_identity(nz_Ez), ddy_HxEz, _identity(nx_Ez))
    Dy_HzEx = _kron_zyx(_identity(nz_Ex), ddy_HzEx, _identity(nx_Ex))
    Dz_HyEx = _kron_zyx(ddz_HyEx, _identity(ny_Ex), _identity(nx_Ex))
    Dz_HxEy = _kron_zyx(ddz_HxEy, _identity(ny_Ey), _identity(nx_Ey))

    Dx_EyHz = _kron_zyx(_identity(nz_Ey), _identity(ny_Ey), ddx_EyHz)
    Dx_EzHy = _kron_zyx(_identity(nz_Ez), _identity(ny_Ez), ddx_EzHy)
    Dy_EzHx = _kron_zyx(_identity(nz_Ez), ddy_EzHx, _identity(nx_Ez))
    Dy_ExHz = _kron_zyx(_identity(nz_Ex), ddy_ExHz, _identity(nx_Ex))
    Dz_ExHy = _kron_zyx(ddz_ExHy, _identity(ny_Ex), _identity(nx_Ex))
    Dz_EyHx = _kron_zyx(ddz_EyHx, _identity(ny_Ey), _identity(nx_Ey))

    curl_E = sparse.bmat(
        [
            [_zero(nt_Hx, nt_Ex), -Dz_HxEy, Dy_HxEz],
            [Dz_HyEx, _zero(nt_Hy, nt_Ey), -Dx_HyEz],
            [-Dy_HzEx, Dx_HzEy, _zero(nt_Hz, nt_Ez)],
        ],
        format="csc",
    )
    curl_H = sparse.bmat(
        [
            [_zero(nt_Ex, nt_Hx), -Dz_ExHy, Dy_ExHz],
            [Dz_EyHx, _zero(nt_Ey, nt_Hy), -Dx_EyHz],
            [-Dy_EzHx, Dx_EzHy, _zero(nt_Ez, nt_Hz)],
        ],
        format="csc",
    )

    # Electric unknowns are stacked as [Ex[:]; Ey[:]; Ez[:]] using Julia's
    # column-major order, so x is the fastest-varying coordinate inside each
    # component block.
    epsilon_diagonal = _diag(
        np.concatenate(
            [
                epsilon_xx.ravel(order="F"),
                epsilon_yy.ravel(order="F"),
                epsilon_zz.ravel(order="F"),
            ]
        )
    )
    A = curl_H @ curl_E - (k0dx**2) * epsilon_diagonal

    if include_off_diagonal:
        # Off-diagonal tensor terms live on Yee lower-corner grids.  Julia
        # averages the source component onto that corner grid, multiplies by
        # epsilon_ij, then averages the result back onto the target component.
        avg_x_Ex_h = avg_x_Ex.conjugate().transpose().tocsc()
        avg_y_Ey_h = avg_y_Ey.conjugate().transpose().tocsc()
        avg_z_Ez_h = avg_z_Ez.conjugate().transpose().tocsc()
        zero_xx = _zero(nt_Ex, nt_Ex)
        zero_yy = _zero(nt_Ey, nt_Ey)
        zero_zz = _zero(nt_Ez, nt_Ez)
        block_xy = (
            _off_diagonal_material_block(
                _kron_zyx(_identity(nz_Ex), _identity(ny_Ex), avg_x_Ex_h),
                epsilon_xy,
                _kron_zyx(_identity(nz_Ey), avg_y_Ey, _identity(nx_Ey)),
                k0dx,
            )
            if epsilon_xy is not None
            else _zero(nt_Ex, nt_Ey)
        )
        block_xz = (
            _off_diagonal_material_block(
                _kron_zyx(_identity(nz_Ex), _identity(ny_Ex), avg_x_Ex_h),
                epsilon_xz,
                _kron_zyx(avg_z_Ez, _identity(ny_Ez), _identity(nx_Ez)),
                k0dx,
            )
            if epsilon_xz is not None
            else _zero(nt_Ex, nt_Ez)
        )
        block_yx = (
            _off_diagonal_material_block(
                _kron_zyx(_identity(nz_Ey), avg_y_Ey_h, _identity(nx_Ey)),
                epsilon_yx,
                _kron_zyx(_identity(nz_Ex), _identity(ny_Ex), avg_x_Ex),
                k0dx,
            )
            if epsilon_yx is not None
            else _zero(nt_Ey, nt_Ex)
        )
        block_yz = (
            _off_diagonal_material_block(
                _kron_zyx(_identity(nz_Ey), avg_y_Ey_h, _identity(nx_Ey)),
                epsilon_yz,
                _kron_zyx(avg_z_Ez, _identity(ny_Ez), _identity(nx_Ez)),
                k0dx,
            )
            if epsilon_yz is not None
            else _zero(nt_Ey, nt_Ez)
        )
        block_zx = (
            _off_diagonal_material_block(
                _kron_zyx(avg_z_Ez_h, _identity(ny_Ez), _identity(nx_Ez)),
                epsilon_zx,
                _kron_zyx(_identity(nz_Ex), _identity(ny_Ex), avg_x_Ex),
                k0dx,
            )
            if epsilon_zx is not None
            else _zero(nt_Ez, nt_Ex)
        )
        block_zy = (
            _off_diagonal_material_block(
                _kron_zyx(avg_z_Ez_h, _identity(ny_Ez), _identity(nx_Ez)),
                epsilon_zy,
                _kron_zyx(_identity(nz_Ey), avg_y_Ey, _identity(nx_Ey)),
                k0dx,
            )
            if epsilon_zy is not None
            else _zero(nt_Ez, nt_Ey)
        )
        A = A + sparse.bmat(
            [
                [zero_xx, block_xy, block_xz],
                [block_yx, zero_yy, block_yz],
                [block_zx, block_zy, zero_zz],
            ],
            format="csc",
        )

    if use_UPML:
        sx_Ex = sx_Hz
        sy_Ey = sy_Hx
        sz_Ez = sz_Hy
        sy_Ex = sy_Ez
        sz_Ey = sz_Ex
        sx_Ez = sx_Ey

        sx_Ex_3d, sy_Ex_3d, sz_Ex_3d = np.meshgrid(sx_Ex, sy_Ex, sz_Ex, indexing="ij")
        sx_Ey_3d, sy_Ey_3d, sz_Ey_3d = np.meshgrid(sx_Ey, sy_Ey, sz_Ey, indexing="ij")
        sx_Ez_3d, sy_Ez_3d, sz_Ez_3d = np.meshgrid(sx_Ez, sy_Ez, sz_Ez, indexing="ij")
        S_E = _diag(
            np.concatenate(
                [
                    (sy_Ex_3d * sz_Ex_3d / sx_Ex_3d).ravel(order="F"),
                    (sx_Ey_3d * sz_Ey_3d / sy_Ey_3d).ravel(order="F"),
                    (sx_Ez_3d * sy_Ez_3d / sz_Ez_3d).ravel(order="F"),
                ]
            )
        )
        A = S_E @ A

    expected_shape = (nt_Ex + nt_Ey + nt_Ez, nt_Ex + nt_Ey + nt_Ez)
    if A.shape != expected_shape:
        raise RuntimeError(f"Internal 3D FDFD matrix shape error: got {A.shape}, expected {expected_shape}")

    is_symmetric_A = True
    if (
        _bloch_breaks_symmetry_3d(xBC_original, epsilon_xx.shape)
        or _bloch_breaks_symmetry_3d(yBC_original, epsilon_yy.shape)
        or _bloch_breaks_symmetry_3d(zBC_original, epsilon_zz.shape)
    ):
        is_symmetric_A = False
    elif _any_pml(xPML) or _any_pml(yPML) or _any_pml(zPML):
        is_symmetric_A = False

    return A.tocsc(), is_symmetric_A, xPML, yPML, zPML


def mesti_build_fdfd_matrix(
    epsilon_xx: np.ndarray,
    k0dx: float | complex,
    yBC: str | numbers.Number,
    zBC: str | numbers.Number,
    yPML: Sequence[PML] | None = None,
    zPML: Sequence[PML] | None = None,
    use_UPML: bool = True,
    *,
    epsilon_yy: np.ndarray | None = None,
    epsilon_zz: np.ndarray | None = None,
    epsilon_xy: np.ndarray | None = None,
    epsilon_xz: np.ndarray | None = None,
    epsilon_yx: np.ndarray | None = None,
    epsilon_yz: np.ndarray | None = None,
    epsilon_zx: np.ndarray | None = None,
    epsilon_zy: np.ndarray | None = None,
    xBC: str | numbers.Number | None = None,
    xPML: Sequence[PML] | None = None,
) -> tuple[sparse.csc_matrix, bool, list[PML], list[PML]] | tuple[sparse.csc_matrix, bool, list[PML], list[PML], list[PML]]:
    """Build a Julia-compatible FDFD operator.

    2D ``epsilon_xx`` inputs build the verified TM/scalar operator and return
    ``(A, is_symmetric_A, yPML, zPML)``.  3D inputs cover vectorial tensor
    permittivity, including optional off-diagonal components, and return
    ``(A, is_symmetric_A, xPML, yPML, zPML)``.  Both paths use Julia-compatible
    column-major vectorization.
    """

    epsilon_xx = np.asarray(epsilon_xx, dtype=np.complex128)
    off_diagonal = (epsilon_xy, epsilon_xz, epsilon_yx, epsilon_yz, epsilon_zx, epsilon_zy)
    if epsilon_xx.ndim == 3:
        if epsilon_yy is None or epsilon_zz is None:
            raise ValueError("3D vectorial assembly requires epsilon_yy and epsilon_zz.")
        if xBC is None:
            raise ValueError("3D vectorial assembly requires xBC.")
        return _mesti_build_fdfd_matrix_3d_diagonal(
            epsilon_xx,
            epsilon_yy,
            epsilon_zz,
            k0dx,
            xBC,
            yBC,
            zBC,
            xPML,
            yPML,
            zPML,
            use_UPML,
            epsilon_xy=epsilon_xy,
            epsilon_xz=epsilon_xz,
            epsilon_yx=epsilon_yx,
            epsilon_yz=epsilon_yz,
            epsilon_zx=epsilon_zx,
            epsilon_zy=epsilon_zy,
        )

    if epsilon_xx.ndim == 2 and (
        epsilon_yy is not None
        or epsilon_zz is not None
        or any(component is not None for component in off_diagonal)
    ):
        raise ValueError("Only epsilon_xx is accepted for 2D TM/scalar assembly.")
    if epsilon_xx.ndim != 2:
        raise NotImplementedError("Only 2D TM/scalar and 3D vectorial FDFD assembly are ported so far.")
    ny_Ex, nz_Ex = epsilon_xx.shape
    if ny_Ex <= 0 or nz_Ex <= 0:
        raise ValueError("epsilon_xx must have positive ny and nz dimensions.")

    yPML = _pml_pair(yPML)
    zPML = _pml_pair(zPML)
    yBC_original = yBC
    zBC_original = zBC
    yBC = convert_BC(yBC, "y")
    zBC = convert_BC(zBC, "z")

    epsilon_bg_y = [1.0, 1.0]
    epsilon_bg_z = [1.0, 1.0]
    if _any_pml(yPML):
        # Julia derives default PML strengths from the boundary-adjacent
        # background permittivity; periodic sides use the averaged value on both
        # ends so the coordinate stretch is continuous across the boundary.
        epsilon_bg_y = [
            float(np.real(np.mean(epsilon_xx[0, :]))),
            float(np.real(np.mean(epsilon_xx[-1, :]))),
        ]
        if _is_number(yBC) or yBC == "periodic":
            mean_bg = float(np.mean(epsilon_bg_y))
            epsilon_bg_y = [mean_bg, mean_bg]
    if _any_pml(zPML):
        # In the padded mesti2s system these z-edge means are taken from the
        # homogeneous spacer/PML regions, matching Julia's call into
        # mesti_build_fdfd_matrix after adding epsilon_low/epsilon_high slabs.
        epsilon_bg_z = [
            float(np.real(np.mean(epsilon_xx[:, 0]))),
            float(np.real(np.mean(epsilon_xx[:, -1]))),
        ]
        if _is_number(zBC) or zBC == "periodic":
            mean_bg = float(np.mean(epsilon_bg_z))
            epsilon_bg_z = [mean_bg, mean_bg]

    yPML = mesti_set_PML_params(yPML, k0dx, epsilon_bg_y, "y")
    zPML = mesti_set_PML_params(zPML, k0dx, epsilon_bg_z, "z")

    ddy_E, _, sy_E, sy_H, _ = build_ddx_E(ny_Ex, yBC, yPML, "y")
    ddz_E, _, sz_E, sz_H, _ = build_ddx_E(nz_Ex, zBC, zPML, "z")

    ddy_H = (-ddy_E.conjugate().transpose()).tocsc()
    ddz_H = (-ddz_E.conjugate().transpose()).tocsc()

    ddy_E = _diag(1 / sy_H) @ ddy_E
    ddz_E = _diag(1 / sz_H) @ ddz_E

    if not use_UPML:
        ddy_H = _diag(1 / sy_E) @ ddy_H
        ddz_H = _diag(1 / sz_E) @ ddz_H

    nt_Ex = ny_Ex * nz_Ex
    if not use_UPML:
        A = (
            -sparse.kron(ddz_H @ ddz_E, _identity(ny_Ex), format="csc")
            - sparse.kron(_identity(nz_Ex), ddy_H @ ddy_E, format="csc")
            # Julia's epsilon_xx[:] is column-major, so y is the fastest
            # varying index in the diagonal material term.
            - _diag((k0dx**2) * epsilon_xx.ravel(order="F"))
        )
    else:
        syz_E = sy_E.reshape(-1, 1) * sz_E.reshape(1, -1)
        A = (
            -sparse.kron(ddz_H @ ddz_E, _diag(sy_E), format="csc")
            - sparse.kron(_diag(sz_E), ddy_H @ ddy_E, format="csc")
            # Keep the same column-major flattening for the UPML-scaled material
            # term as Julia's syz_E[:].*epsilon_xx[:] expression.
            - _diag((k0dx**2) * (syz_E * epsilon_xx).ravel(order="F"))
        )

    if A.shape != (nt_Ex, nt_Ex):
        raise RuntimeError(f"Internal FDFD matrix shape error: got {A.shape}, expected {(nt_Ex, nt_Ex)}")

    is_symmetric_A = True
    if _bloch_breaks_symmetry(yBC_original, ny_Ex) or _bloch_breaks_symmetry(zBC_original, nz_Ex):
        is_symmetric_A = False
    elif (not use_UPML) and (_any_pml(yPML) or _any_pml(zPML)):
        is_symmetric_A = False

    return A.tocsc(), is_symmetric_A, yPML, zPML
