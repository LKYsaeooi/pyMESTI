"""High-level MESTI solve wrapper."""

from __future__ import annotations

from copy import deepcopy
import numbers
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from .fdfd_matrix import mesti_build_fdfd_matrix
from .solver import mesti_matrix_solver
from .types import Info, Matrices, Opts, PML, Source_struct, Syst


def _default_pml_pair(direction: str | None = None) -> list[PML]:
    return [PML(0, direction=direction, side="-"), PML(0, direction=direction, side="+")]


def _clone_pml_layer(layer: PML, direction: str, side: str) -> PML:
    copied = deepcopy(layer)
    if copied.npixels is None:
        copied.npixels = 0
    copied.direction = direction
    copied.side = side
    return copied


def _pml_list(value: Any) -> list[PML]:
    if value is None:
        return []
    pml = value if isinstance(value, list) else [value]
    if not all(isinstance(layer, PML) for layer in pml):
        raise TypeError("syst.PML must be a PML object or a list of PML objects.")
    return list(pml)


def _legacy_undirected_pml_pairs(
    pml: Sequence[PML],
    dimensions: Sequence[str],
) -> tuple[list[PML], ...] | None:
    if len(pml) != 2 or any(layer.direction is not None for layer in pml):
        return None
    if tuple(dimensions) == ("y", "z"):
        return (
            _default_pml_pair("y"),
            [_clone_pml_layer(pml[0], "z", "-"), _clone_pml_layer(pml[1], "z", "+")],
        )
    pair_by_dimension = tuple(
        [_clone_pml_layer(pml[0], direction, "-"), _clone_pml_layer(pml[1], direction, "+")]
        for direction in dimensions
    )
    return pair_by_dimension


def _parse_direct_pml_pairs(syst: Syst, dimensions: Sequence[str]) -> tuple[list[PML], ...]:
    pml = _pml_list(syst.PML)
    if not pml:
        return tuple(_default_pml_pair(direction) for direction in dimensions)

    legacy = _legacy_undirected_pml_pairs(pml, dimensions)
    if legacy is not None:
        return legacy

    slots: dict[str, list[PML | None]] = {direction: [None, None] for direction in dimensions}
    dimension_set = set(dimensions)
    valid_directions = {"all", "x", "y", "z", "xy", "xz", "yz", "yx", "zx", "zy"}
    valid_sides = {"both", "-", "+"}
    for index, layer in enumerate(pml, start=1):
        direction_label = "all" if layer.direction is None else str(layer.direction).lower()
        if direction_label not in valid_directions:
            raise ValueError(
                f'syst.PML[{index}].direction = "{layer.direction}" is not supported; '
                'use "all", "x", "y", "z", "xy", "xz", "yz", "yx", "zx", or "zy".'
            )
        side_label = "both" if layer.side is None else str(layer.side).lower()
        if side_label not in valid_sides:
            raise ValueError(f'syst.PML[{index}].side = "{layer.side}" is not supported; use "both", "-", or "+".')

        target_directions = list(dimensions) if direction_label == "all" else [
            direction for direction in direction_label if direction in dimension_set
        ]
        target_sides = [("-", 0), ("+", 1)] if side_label == "both" else [(side_label, 0 if side_label == "-" else 1)]
        for direction in target_directions:
            for side, side_index in target_sides:
                if slots[direction][side_index] is not None:
                    raise ValueError(f"PML on {side}{direction} side is specified more than once in syst.PML.")
                slots[direction][side_index] = _clone_pml_layer(layer, direction, side)

    pairs: list[list[PML]] = []
    for direction in dimensions:
        low, high = slots[direction]
        pairs.append(
            [
                low if low is not None else PML(0, direction=direction, side="-"),
                high if high is not None else PML(0, direction=direction, side="+"),
            ]
        )
    return tuple(pairs)


def _pml_by_direction(syst: Syst) -> tuple[list[PML], list[PML]]:
    """Extract y/z PML pairs for the current 2D wrapper."""

    yPML, zPML = _parse_direct_pml_pairs(syst, ("y", "z"))
    return yPML, zPML


def _pml_by_direction_3d(syst: Syst) -> tuple[list[PML], list[PML], list[PML]]:
    """Extract x/y/z PML pairs for direct 3D ``mesti`` calls."""

    xPML, yPML, zPML = _parse_direct_pml_pairs(syst, ("x", "y", "z"))
    return xPML, yPML, zPML


def _is_number(value: object) -> bool:
    return isinstance(value, numbers.Number)


def _resolve_bloch_or_boundary(
    boundary: Any,
    wave_number: Any,
    period_pixels: int,
    dx: float,
    direction: str,
) -> str | numbers.Number:
    if wave_number is not None:
        if boundary is not None and not (isinstance(boundary, str) and boundary.lower() == "bloch"):
            raise ValueError(f'When syst.k{direction}_B is given, syst.{direction}BC must be "Bloch" if specified.')
        scalar = np.asarray(wave_number).reshape(-1)
        if scalar.size != 1:
            raise ValueError(f"syst.k{direction}_B must be a scalar when provided.")
        return scalar[0].item() * period_pixels * dx
    if boundary is None:
        return "PEC"
    if isinstance(boundary, str) and boundary.lower() == "bloch":
        raise ValueError(f'syst.{direction}BC = "Bloch" but syst.k{direction}_B is not given.')
    return boundary


def _direct_boundaries_2d(syst: Syst, epsilon_xx: np.ndarray) -> tuple[str | numbers.Number, str | numbers.Number]:
    ny, nz = epsilon_xx.shape
    return (
        _resolve_bloch_or_boundary(syst.yBC, syst.ky_B, ny, float(syst.dx), "y"),
        _resolve_bloch_or_boundary(syst.zBC, syst.kz_B, nz, float(syst.dx), "z"),
    )


def _direct_boundaries_3d(
    syst: Syst,
    epsilon_xx: np.ndarray,
    epsilon_yy: np.ndarray,
    epsilon_zz: np.ndarray,
) -> tuple[str | numbers.Number, str | numbers.Number, str | numbers.Number]:
    return (
        _resolve_bloch_or_boundary(syst.xBC, syst.kx_B, epsilon_xx.shape[0], float(syst.dx), "x"),
        _resolve_bloch_or_boundary(syst.yBC, syst.ky_B, epsilon_yy.shape[1], float(syst.dx), "y"),
        _resolve_bloch_or_boundary(syst.zBC, syst.kz_B, epsilon_zz.shape[2], float(syst.dx), "z"),
    )


def _use_upml(syst: Syst) -> bool:
    if syst.PML_type is None:
        return True
    pml_type = str(syst.PML_type).lower()
    if pml_type == "upml":
        return True
    if pml_type in {"sc-pml", "scpml"}:
        return False
    raise ValueError(f'syst.PML_type = "{syst.PML_type}" is not supported; use "UPML" or "SC-PML".')


def _bool_option(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"opts.{name} must be a boolean scalar, if given.")


def _prefactor(opts: Opts) -> complex:
    if opts.prefactor is None:
        opts.prefactor = 1
        return 1
    arr = np.asarray(opts.prefactor)
    if arr.size != 1:
        raise ValueError("opts.prefactor must be a numeric scalar, if given.")
    scalar = arr.reshape(-1)[0]
    if not _is_number(scalar):
        raise ValueError("opts.prefactor must be a numeric scalar, if given.")
    return scalar.item() if hasattr(scalar, "item") else scalar


def _prepare_direct_options(opts: Opts, return_field_profile: bool) -> complex:
    opts.return_field_profile = return_field_profile
    prefactor = _prefactor(opts)
    if return_field_profile:
        if opts.exclude_PML_in_field_profiles is None:
            opts.exclude_PML_in_field_profiles = False
        else:
            opts.exclude_PML_in_field_profiles = _bool_option(
                opts.exclude_PML_in_field_profiles,
                "exclude_PML_in_field_profiles",
            )
    elif opts.exclude_PML_in_field_profiles is not None:
        opts.exclude_PML_in_field_profiles = None
    return prefactor


def _is_transpose_b_projection(value: Any) -> bool:
    return isinstance(value, str) and value.replace(" ", "").lower() == "transpose(b)"


def _dense_like_matrix(matrix: Any, name: str) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.complex128, copy=False)
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    return arr


def _finalize_projection_result(S: np.ndarray, D: Any, prefactor: complex) -> np.ndarray:
    result = prefactor * S
    if D is None:
        return result
    D_arr = _dense_like_matrix(D, "D")
    if D_arr.shape != result.shape:
        raise ValueError("D shape must match projected solution shape.")
    return result - D_arr


def _trim_2d_field_profile(Ex: np.ndarray, yPML: Sequence[PML], zPML: Sequence[PML]) -> np.ndarray:
    y0 = yPML[0].npixels or 0
    y1 = Ex.shape[0] - (yPML[1].npixels or 0)
    z0 = zPML[0].npixels or 0
    z1 = Ex.shape[1] - (zPML[1].npixels or 0)
    return Ex[y0:y1, z0:z1, :]


def _trim_3d_field_profile(
    field: np.ndarray,
    xPML: Sequence[PML],
    yPML: Sequence[PML],
    zPML: Sequence[PML],
) -> np.ndarray:
    x0 = xPML[0].npixels or 0
    x1 = field.shape[0] - (xPML[1].npixels or 0)
    y0 = yPML[0].npixels or 0
    y1 = field.shape[1] - (yPML[1].npixels or 0)
    z0 = zPML[0].npixels or 0
    z1 = field.shape[2] - (zPML[1].npixels or 0)
    return field[x0:x1, y0:y1, z0:z1, :]


def _as_2d_rhs(matrix: Any, rows: int, name: str) -> Any:
    if sparse.issparse(matrix):
        if matrix.shape[0] != rows:
            raise ValueError(f"{name} row count must be {rows}")
        return matrix.tocsc()
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2 or arr.shape[0] != rows:
        raise ValueError(f"{name} must have shape ({rows}, nrhs)")
    return arr


def _as_projection_matrix(matrix: Any, cols: int, name: str) -> Any:
    if sparse.issparse(matrix):
        if matrix.shape[1] != cols:
            raise ValueError(f"{name} column count must be {cols}")
        return matrix.tocsc()
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != cols:
        raise ValueError(f"{name} must have shape (noutputs, {cols})")
    return arr


def _source_items(source: Source_struct) -> tuple[list[Any], list[Any], str]:
    if source.isempty:
        raise ValueError("Empty Source_struct cannot be converted to a matrix here.")
    if source.ind is not None and source.pos is not None:
        raise ValueError("Source_struct cannot contain both ind and pos.")
    if source.ind is not None:
        if source.data is None:
            raise ValueError("Source_struct with ind must contain data.")
        specs = list(source.ind)
        data = list(source.data)
        if len(specs) != len(data):
            raise ValueError("Source_struct ind and data must have the same length.")
        return specs, data, "ind"
    if source.pos is not None:
        if source.data is None:
            raise ValueError("Source_struct with pos must contain data.")
        specs = list(source.pos)
        data = list(source.data)
        if len(specs) != len(data):
            raise ValueError("Source_struct pos and data must have the same length.")
        return specs, data, "pos"
    raise ValueError("Source_struct must contain ind or pos.")


def _coerce_data_columns(data: Any, n_locations: int) -> np.ndarray:
    arr = np.asarray(data, dtype=np.complex128)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim > 2:
        # Julia reshapes Source_struct data in column-major order before
        # sparse insertion; keep the same ordering for 2D TM parity.
        arr = arr.reshape((n_locations, -1), order="F")
    if arr.ndim != 2 or arr.shape[0] != n_locations:
        raise ValueError("Source_struct data shape does not match specified locations.")
    return arr


def _source_struct_is_empty(source: Source_struct, name: str = "Source_struct") -> bool:
    isempty = False if source.isempty is None else source.isempty
    if not isinstance(isempty, (bool, np.bool_)):
        raise ValueError(f"{name}.isempty must be a boolean scalar when provided.")
    if bool(isempty) and (source.pos is not None or source.ind is not None or source.data is not None):
        raise ValueError(f"{name}.isempty=True cannot be combined with pos, ind, or data.")
    return bool(isempty)


def _source_struct_to_matrix(
    source: Source_struct,
    shape_yz: tuple[int, int],
    for_projection: bool = False,
) -> sparse.csc_matrix:
    ny, nz = shape_yz
    total = ny * nz
    specs, data_blocks, mode = _source_items(source)

    blocks: list[sparse.csc_matrix] = []

    for spec, data in zip(specs, data_blocks):
        if mode == "ind":
            indices = np.asarray(spec, dtype=int).reshape(-1)
        else:
            pos = np.asarray(spec, dtype=int).reshape(-1)
            if pos.size != 4:
                raise ValueError("2D Source_struct pos entries must be [y1, z1, y2, z2].")
            y1, z1, y2, z2 = pos.tolist()
            if y1 < 0 or z1 < 0 or y2 >= ny or z2 >= nz or y2 < y1 or z2 < z1:
                raise ValueError("2D Source_struct pos is outside the field grid.")
            yy, zz = np.meshgrid(
                np.arange(y1, y2 + 1),
                np.arange(z1, z2 + 1),
                indexing="ij",
            )
            # Source_struct.pos is zero-based and inclusive in Python, but it
            # still flattens as Julia does: y varies fastest, then z.
            indices = (yy + zz * ny).reshape(-1, order="F")

        if np.any(indices < 0) or np.any(indices >= total):
            raise ValueError("Source_struct indices must be zero-based and inside the field grid.")
        data_arr = _coerce_data_columns(data, len(indices))

        cols = np.repeat(np.arange(data_arr.shape[1]), len(indices))
        rows = np.tile(indices, data_arr.shape[1])
        # Flatten data columns in the same order as the column-major location
        # list above so source/projection blocks match Julia exactly.
        values = data_arr.reshape(-1, order="F")
        block = sparse.coo_matrix(
            (values, (rows, cols)),
            shape=(total, data_arr.shape[1]),
            dtype=np.complex128,
        ).tocsc()
        blocks.append(block.transpose().tocsc() if for_projection else block)

    if not blocks:
        raise ValueError("Source_struct contains no data blocks.")

    if for_projection:
        return sparse.vstack(blocks, format="csc")
    return sparse.hstack(blocks, format="csc")


def _source_struct_to_matrix_3d(
    source: Source_struct,
    shape_xyz: tuple[int, int, int],
    for_projection: bool = False,
) -> sparse.csc_matrix:
    nx, ny, nz = shape_xyz
    total = nx * ny * nz
    specs, data_blocks, mode = _source_items(source)

    blocks: list[sparse.csc_matrix] = []

    for spec, data in zip(specs, data_blocks):
        if mode == "ind":
            indices = np.asarray(spec, dtype=int).reshape(-1)
        else:
            pos = np.asarray(spec, dtype=int).reshape(-1)
            if pos.size != 6:
                raise ValueError("3D Source_struct pos entries must be [x1, y1, z1, x2, y2, z2].")
            x1, y1, z1, x2, y2, z2 = pos.tolist()
            if x1 < 0 or y1 < 0 or z1 < 0 or x2 >= nx or y2 >= ny or z2 >= nz or x2 < x1 or y2 < y1 or z2 < z1:
                raise ValueError("3D Source_struct pos is outside the component field grid.")
            xx, yy, zz = np.meshgrid(
                np.arange(x1, x2 + 1),
                np.arange(y1, y2 + 1),
                np.arange(z1, z2 + 1),
                indexing="ij",
            )
            # Python keeps Source_struct.pos zero-based and inclusive.  The
            # linearization remains Julia-compatible for a 3D component block:
            # x varies fastest, then y, then z.
            indices = (xx + yy * nx + zz * nx * ny).reshape(-1, order="F")

        if np.any(indices < 0) or np.any(indices >= total):
            raise ValueError("Source_struct indices must be zero-based and inside the component field grid.")
        data_arr = _coerce_data_columns(data, len(indices))

        cols = np.repeat(np.arange(data_arr.shape[1]), len(indices))
        rows = np.tile(indices, data_arr.shape[1])
        values = data_arr.reshape(-1, order="F")
        block = sparse.coo_matrix(
            (values, (rows, cols)),
            shape=(total, data_arr.shape[1]),
            dtype=np.complex128,
        ).tocsc()
        blocks.append(block.transpose().tocsc() if for_projection else block)

    if not blocks:
        raise ValueError("Source_struct contains no data blocks.")

    if for_projection:
        return sparse.vstack(blocks, format="csc")
    return sparse.hstack(blocks, format="csc")


def _component_sizes(shapes_xyz: Sequence[tuple[int, int, int]]) -> tuple[int, ...]:
    return tuple(int(np.prod(shape)) for shape in shapes_xyz)


def _source_struct_sequence_to_matrix_3d(
    value: Sequence[Source_struct],
    shapes_xyz: Sequence[tuple[int, int, int]],
    name: str,
    for_projection: bool = False,
) -> sparse.csc_matrix:
    if len(value) != 3 or not all(isinstance(item, Source_struct) for item in value):
        raise ValueError(f"3D {name} Source_struct input must contain exactly three component Source_struct objects.")

    component_sizes = _component_sizes(shapes_xyz)
    matrices: list[sparse.csc_matrix | None] = []
    shared_count: int | None = None
    for component_index, (source, shape_xyz) in enumerate(zip(value, shapes_xyz), start=1):
        if _source_struct_is_empty(source, f"{name}[{component_index}]"):
            matrices.append(None)
            continue
        matrix = _source_struct_to_matrix_3d(source, shape_xyz, for_projection=for_projection)
        count = matrix.shape[0] if for_projection else matrix.shape[1]
        if shared_count is None:
            shared_count = count
        elif count != shared_count:
            role = "projection rows" if for_projection else "RHS columns"
            raise ValueError(f"3D {name} components must have the same number of {role}.")
        matrices.append(matrix)

    if shared_count is None:
        raise ValueError(f"3D {name} Source_struct input contains no active component.")

    filled: list[sparse.csc_matrix] = []
    for matrix, component_size in zip(matrices, component_sizes):
        if matrix is not None:
            filled.append(matrix)
        elif for_projection:
            filled.append(sparse.csc_matrix((shared_count, component_size), dtype=np.complex128))
        else:
            filled.append(sparse.csc_matrix((component_size, shared_count), dtype=np.complex128))

    # Julia combines component sources as B=[Bx; By; Bz] and component
    # projections as C=[Cx Cy Cz], matching the electric-field stack
    # [Ex[:]; Ey[:]; Ez[:]] used by the 3D FDFD matrix.
    return sparse.hstack(filled, format="csc") if for_projection else sparse.vstack(filled, format="csc")


def _source_like_to_matrix(value: Any, shape_yz: tuple[int, int], name: str, for_projection: bool = False) -> Any:
    total = shape_yz[0] * shape_yz[1]
    if isinstance(value, Source_struct):
        return _source_struct_to_matrix(value, shape_yz, for_projection=for_projection)
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, Source_struct) for item in value):
        active = [item for item in value if not item.isempty]
        if not active:
            raise ValueError(f"{name} Source_struct list contains no active source.")
        return _source_struct_to_matrix(active[0], shape_yz, for_projection=for_projection)
    if for_projection:
        return _as_projection_matrix(value, total, name)
    return _as_2d_rhs(value, total, name)


def _source_like_to_matrix_3d(
    value: Any,
    shapes_xyz: Sequence[tuple[int, int, int]],
    name: str,
    for_projection: bool = False,
) -> Any:
    total = sum(_component_sizes(shapes_xyz))
    if isinstance(value, Source_struct):
        raise ValueError(f"3D {name} Source_struct input must be a three-component sequence.")
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, Source_struct) for item in value):
        return _source_struct_sequence_to_matrix_3d(value, shapes_xyz, name, for_projection=for_projection)
    if for_projection:
        return _as_projection_matrix(value, total, name)
    return _as_2d_rhs(value, total, name)


def _off_diagonal_components(syst: Syst) -> tuple[Any, ...]:
    return (syst.epsilon_xy, syst.epsilon_xz, syst.epsilon_yx, syst.epsilon_yz, syst.epsilon_zx, syst.epsilon_zy)


def _mesti_3d_diagonal(
    syst: Syst,
    epsilon_xx: np.ndarray,
    B: Any,
    C: Any,
    D: Any,
    opts: Opts,
    prefactor: complex,
) -> tuple[Any, ...]:
    if syst.epsilon_yy is None or syst.epsilon_zz is None:
        raise ValueError("3D direct mesti requires syst.epsilon_yy and syst.epsilon_zz.")

    epsilon_yy = np.asarray(syst.epsilon_yy, dtype=np.complex128)
    epsilon_zz = np.asarray(syst.epsilon_zz, dtype=np.complex128)
    if epsilon_yy.ndim != 3 or epsilon_zz.ndim != 3:
        raise ValueError("3D direct mesti requires 3D epsilon_yy and epsilon_zz arrays.")
    epsilon_xy, epsilon_xz, epsilon_yx, epsilon_yz, epsilon_zx, epsilon_zy = (
        None if component is None else np.asarray(component, dtype=np.complex128)
        for component in _off_diagonal_components(syst)
    )

    k0dx = (2 * np.pi / syst.wavelength) * syst.dx
    xBC, yBC, zBC = _direct_boundaries_3d(syst, epsilon_xx, epsilon_yy, epsilon_zz)
    xPML, yPML, zPML = _pml_by_direction_3d(syst)
    use_UPML = _use_upml(syst)
    # Off-diagonal tensor components use the Yee lower-corner shapes validated
    # in the low-level matrix builder; direct mesti keeps the same E-field
    # stack and source/projection conventions as the diagonal 3D path.
    A, is_symmetric_A, xPML, yPML, zPML = mesti_build_fdfd_matrix(
        epsilon_xx,
        k0dx,
        yBC,
        zBC,
        yPML,
        zPML,
        use_UPML=use_UPML,
        epsilon_yy=epsilon_yy,
        epsilon_zz=epsilon_zz,
        epsilon_xy=epsilon_xy,
        epsilon_xz=epsilon_xz,
        epsilon_yx=epsilon_yx,
        epsilon_yz=epsilon_yz,
        epsilon_zx=epsilon_zx,
        epsilon_zy=epsilon_zy,
        xBC=xBC,
        xPML=xPML,
    )
    if opts.is_symmetric_A is None:
        opts.is_symmetric_A = bool(is_symmetric_A)

    shapes_xyz = (epsilon_xx.shape, epsilon_yy.shape, epsilon_zz.shape)
    B_matrix = _source_like_to_matrix_3d(B, shapes_xyz, "B")
    if C is None:
        X, info = mesti_matrix_solver(Matrices(A=A, B=B_matrix), opts)
        info.xPML = xPML
        info.yPML = yPML
        info.zPML = zPML
        info.is_symmetric_A = is_symmetric_A

        nt_Ex, nt_Ey, nt_Ez = _component_sizes(shapes_xyz)
        # Split the solver result along the same component offsets used to
        # build A and B.  Each component is then reshaped in Fortran order so
        # Python returns arrays with the same [x, y, z, rhs] layout as Julia.
        X = prefactor * X
        Ex = X[:nt_Ex, :].reshape((*epsilon_xx.shape, X.shape[1]), order="F")
        Ey = X[nt_Ex : nt_Ex + nt_Ey, :].reshape((*epsilon_yy.shape, X.shape[1]), order="F")
        Ez = X[nt_Ex + nt_Ey : nt_Ex + nt_Ey + nt_Ez, :].reshape((*epsilon_zz.shape, X.shape[1]), order="F")
        if opts.exclude_PML_in_field_profiles:
            Ex = _trim_3d_field_profile(Ex, xPML, yPML, zPML)
            Ey = _trim_3d_field_profile(Ey, xPML, yPML, zPML)
            Ez = _trim_3d_field_profile(Ez, xPML, yPML, zPML)
        return Ex, Ey, Ez, info

    C_matrix = (
        "transpose(B)"
        if _is_transpose_b_projection(C)
        else _source_like_to_matrix_3d(C, shapes_xyz, "C", for_projection=True)
    )
    S, info = mesti_matrix_solver(Matrices(A=A, B=B_matrix, C=C_matrix), opts)
    S = _finalize_projection_result(S, D, prefactor)
    info.xPML = xPML
    info.yPML = yPML
    info.zPML = zPML
    info.is_symmetric_A = is_symmetric_A
    return S, info


def mesti(
    syst: Syst,
    B: Any,
    C: Any = None,
    D: Any = None,
    opts: Opts | None = None,
) -> tuple[Any, ...]:
    """Run a direct MESTI solve for supported 2D TM or 3D vectorial systems."""

    if isinstance(C, Opts):
        if D is not None or opts is not None:
            raise TypeError("mesti(syst, B, opts) cannot also receive D or opts.")
        opts = C
        C = None
    elif isinstance(D, Opts):
        if opts is not None:
            raise TypeError("mesti(..., opts) received opts both positionally and by keyword.")
        opts = D
        D = None

    if not isinstance(syst, Syst):
        raise TypeError("syst must be a Syst instance")
    epsilon_xx = np.asarray(syst.epsilon_xx, dtype=np.complex128)
    if syst.wavelength is None or syst.dx is None:
        raise ValueError("syst.wavelength and syst.dx must be set.")
    if C is None and D is not None:
        raise ValueError("D must be None when C is omitted.")
    if isinstance(C, str) and not _is_transpose_b_projection(C):
        raise ValueError('Input argument C must be a numeric matrix, Source_struct sequence, or "transpose(B)".')

    opts = opts if opts is not None else Opts()
    # Julia records whether direct mesti() returned a full field profile or a
    # projected C*inv(A)*B result in info.opts.return_field_profile.  The
    # scalar prefactor is applied before optional D subtraction, matching
    # mesti_main.jl's post-solve ordering.
    prefactor = _prepare_direct_options(opts, C is None)

    if epsilon_xx.ndim == 3:
        return _mesti_3d_diagonal(syst, epsilon_xx, B, C, D, opts, prefactor)
    if epsilon_xx.ndim != 2:
        raise NotImplementedError("Only 2D TM and 3D vectorial direct MESTI solves are ported so far.")

    ny, nz = epsilon_xx.shape
    k0dx = (2 * np.pi / syst.wavelength) * syst.dx
    yBC, zBC = _direct_boundaries_2d(syst, epsilon_xx)
    yPML, zPML = _pml_by_direction(syst)
    use_UPML = _use_upml(syst)
    A, is_symmetric_A, yPML, zPML = mesti_build_fdfd_matrix(
        epsilon_xx,
        k0dx,
        yBC,
        zBC,
        yPML,
        zPML,
        use_UPML=use_UPML,
    )
    if opts.is_symmetric_A is None:
        opts.is_symmetric_A = bool(is_symmetric_A)

    B_matrix = _source_like_to_matrix(B, (ny, nz), "B")
    if C is None:
        X, info = mesti_matrix_solver(Matrices(A=A, B=B_matrix), opts)
        info.yPML = yPML
        info.zPML = zPML
        info.is_symmetric_A = is_symmetric_A
        X = prefactor * X
        Ex = X.reshape((ny, nz, X.shape[1]), order="F")
        if opts.exclude_PML_in_field_profiles:
            Ex = _trim_2d_field_profile(Ex, yPML, zPML)
        return Ex, info

    C_matrix = (
        "transpose(B)"
        if _is_transpose_b_projection(C)
        else _source_like_to_matrix(C, (ny, nz), "C", for_projection=True)
    )
    S, info = mesti_matrix_solver(Matrices(A=A, B=B_matrix, C=C_matrix), opts)
    S = _finalize_projection_result(S, D, prefactor)
    info.yPML = yPML
    info.zPML = zPML
    info.is_symmetric_A = is_symmetric_A
    return S, info
