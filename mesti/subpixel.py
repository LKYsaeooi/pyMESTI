"""Subpixel smoothing helpers for the Python MESTI port."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
import math
from typing import Sequence

import numpy as np


@dataclass
class Cuboid:
    """Axis-aligned rectangle/cuboid geometry compatible with Julia examples."""

    center: Sequence[float]
    widths: Sequence[float]

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        widths = np.asarray(self.widths, dtype=float)
        if center.ndim != 1 or widths.ndim != 1 or center.shape != widths.shape:
            raise ValueError("Cuboid center and widths must be one-dimensional arrays with matching length.")
        if center.size not in (2, 3):
            raise NotImplementedError("Only 2D or 3D Cuboid smoothing is supported in this Python slice.")
        if np.any(widths <= 0):
            raise ValueError("Cuboid widths must be positive.")
        self.center = center
        self.widths = widths

    @property
    def ndim(self) -> int:
        return int(np.asarray(self.center).size)

    @property
    def lower(self) -> np.ndarray:
        return np.asarray(self.center, dtype=float) - np.asarray(self.widths, dtype=float) / 2

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(self.center, dtype=float) + np.asarray(self.widths, dtype=float) / 2

    def translated(self, offset: Sequence[float]) -> "Cuboid":
        return Cuboid(np.asarray(self.center, dtype=float) + np.asarray(offset, dtype=float), self.widths)

    def contains(self, point: Sequence[float], tol: float = 1e-12) -> bool:
        point_arr = np.asarray(point, dtype=float)
        return bool(np.all(point_arr >= self.lower - tol) and np.all(point_arr <= self.upper + tol))


@dataclass
class Ball:
    """Circle/sphere geometry compatible with Julia ``GeometryPrimitives.Ball``."""

    center: Sequence[float]
    radius: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        radius = float(self.radius)
        if center.ndim != 1:
            raise ValueError("Ball center must be a one-dimensional array.")
        if center.size not in (2, 3):
            raise NotImplementedError("Only 2D or 3D Ball compatibility stubs are supported.")
        if radius <= 0:
            raise ValueError("Ball radius must be positive.")
        self.center = center
        self.radius = radius

    @property
    def ndim(self) -> int:
        return int(np.asarray(self.center).size)

    @property
    def lower(self) -> np.ndarray:
        return np.asarray(self.center, dtype=float) - self.radius

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(self.center, dtype=float) + self.radius

    def translated(self, offset: Sequence[float]) -> "Ball":
        return Ball(np.asarray(self.center, dtype=float) + np.asarray(offset, dtype=float), self.radius)

    def contains(self, point: Sequence[float], tol: float = 1e-12) -> bool:
        point_arr = np.asarray(point, dtype=float)
        return bool(np.linalg.norm(point_arr - np.asarray(self.center, dtype=float)) <= self.radius + tol)


def _validate_smoothing_objects(object_list: Sequence[object], ndim: int) -> None:
    if not all(isinstance(obj, (Cuboid, Ball)) and obj.ndim == ndim for obj in object_list):
        raise NotImplementedError("Only 2D or 3D Cuboid objects or Ball objects are supported in this smoothing slice.")


def _convert_bc_sbpsm(BC: str, direction: str) -> str:
    bc = str(BC).lower()
    if bc == "pec":
        return "PEC"
    if bc == "pmc":
        return "PMC"
    if bc == "pecpmc":
        return "PECPMC"
    if bc == "pmcpec":
        return "PMCPEC"
    if bc == "periodic":
        return "periodic"
    if bc == "bloch":
        return "Bloch"
    raise ValueError(f'Input argument {direction}BC = "{BC}" is not a supported option.')


def _normalized_domain_and_objects(
    delta_x: float,
    domain: Cuboid,
    object_list: Sequence[Cuboid],
) -> tuple[Cuboid, list[Cuboid]]:
    lower = domain.lower
    lengths = domain.upper - lower
    adjusted = []
    for axis_length in lengths:
        count = round(axis_length / delta_x)
        if not math.isclose(axis_length / delta_x, count, rel_tol=1e-12, abs_tol=1e-12):
            axis_length = max(count * delta_x, delta_x)
        adjusted.append(axis_length)
    adjusted_lengths = np.asarray(adjusted, dtype=float)
    translated = [obj.translated(-lower) for obj in object_list]
    return Cuboid(adjusted_lengths / 2, adjusted_lengths), translated


def _add_periodic_images(
    domain: Cuboid,
    object_list: list[Cuboid],
    object_epsilon_list: list[complex],
    *boundaries: str,
) -> tuple[list[Cuboid], list[complex]]:
    objects = list(deepcopy(object_list))
    epsilons = list(object_epsilon_list)
    lower = domain.lower
    upper = domain.upper
    period = upper - lower

    for axis, boundary in enumerate(boundaries):
        if boundary not in {"periodic", "Bloch"}:
            continue
        snapshot = list(zip(objects, epsilons))
        for obj, eps_value in snapshot:
            obj_lower = obj.lower
            obj_upper = obj.upper
            if obj_lower[axis] <= lower[axis] <= obj_upper[axis] < upper[axis]:
                offset = np.zeros(domain.ndim)
                offset[axis] = period[axis]
                objects.append(obj.translated(offset))
                epsilons.append(eps_value)
            elif lower[axis] < obj_lower[axis] <= upper[axis] <= obj_upper[axis]:
                offset = np.zeros(domain.ndim)
                offset[axis] = -period[axis]
                objects.append(obj.translated(offset))
                epsilons.append(eps_value)
    return objects, epsilons


def _pick_epsilon_2d_tm(epsilon_xx: np.ndarray, yBC: str, zBC: str) -> np.ndarray:
    if yBC in {"PEC", "PECPMC"}:
        epsilon_xx = epsilon_xx[1:, :]
    if zBC in {"PEC", "PECPMC"}:
        epsilon_xx = epsilon_xx[:, 1:]
    return epsilon_xx


def _pick_inv_epsilon_2d_te(
    inv_epsilon_Ey_site: np.ndarray,
    inv_epsilon_Ez_site: np.ndarray,
    inv_epsilon_Eo_site: np.ndarray,
    yBC: str,
    zBC: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inv_epsilon_yy = inv_epsilon_Ey_site[:, :, 1, 1]
    inv_epsilon_zz = inv_epsilon_Ez_site[:, :, 2, 2]
    inv_epsilon_yz = inv_epsilon_Eo_site[:, :, 1, 2]

    if yBC == "PEC":
        inv_epsilon_yz = inv_epsilon_yz[1:, :]
        inv_epsilon_zz = inv_epsilon_zz[1:, :]
    elif yBC == "PMC":
        inv_epsilon_yy = inv_epsilon_yy[:-1, :]
    elif yBC == "PECPMC":
        inv_epsilon_yy = inv_epsilon_yy[:-1, :]
        inv_epsilon_yz = inv_epsilon_yz[1:, :]
        inv_epsilon_zz = inv_epsilon_zz[1:, :]

    if zBC == "PEC":
        inv_epsilon_yy = inv_epsilon_yy[:, 1:]
        inv_epsilon_yz = inv_epsilon_yz[:, 1:]
    elif zBC == "PMC":
        inv_epsilon_zz = inv_epsilon_zz[:, :-1]
    elif zBC == "PECPMC":
        inv_epsilon_yy = inv_epsilon_yy[:, 1:]
        inv_epsilon_yz = inv_epsilon_yz[:, 1:]
        inv_epsilon_zz = inv_epsilon_zz[:, :-1]

    return inv_epsilon_yy, inv_epsilon_zz, inv_epsilon_yz


def _pick_epsilon_3d(
    epsilon_Eo_site: np.ndarray,
    epsilon_Ex_site: np.ndarray,
    epsilon_Ey_site: np.ndarray,
    epsilon_Ez_site: np.ndarray,
    xBC: str,
    yBC: str,
    zBC: str,
) -> tuple[np.ndarray, ...]:
    epsilon_xx = epsilon_Ex_site[:, :, :, 0, 0]
    epsilon_xy = epsilon_Eo_site[:, :, :, 0, 1]
    epsilon_xz = epsilon_Eo_site[:, :, :, 0, 2]
    epsilon_yx = epsilon_Eo_site[:, :, :, 1, 0]
    epsilon_yy = epsilon_Ey_site[:, :, :, 1, 1]
    epsilon_yz = epsilon_Eo_site[:, :, :, 1, 2]
    epsilon_zx = epsilon_Eo_site[:, :, :, 2, 0]
    epsilon_zy = epsilon_Eo_site[:, :, :, 2, 1]
    epsilon_zz = epsilon_Ez_site[:, :, :, 2, 2]

    if xBC == "PEC":
        epsilon_xy = epsilon_xy[1:, :, :]
        epsilon_xz = epsilon_xz[1:, :, :]
        epsilon_yx = epsilon_yx[1:, :, :]
        epsilon_yy = epsilon_yy[1:, :, :]
        epsilon_yz = epsilon_yz[1:, :, :]
        epsilon_zx = epsilon_zx[1:, :, :]
        epsilon_zy = epsilon_zy[1:, :, :]
        epsilon_zz = epsilon_zz[1:, :, :]
    elif xBC == "PMC":
        epsilon_xx = epsilon_xx[:-1, :, :]
    elif xBC == "PECPMC":
        epsilon_xx = epsilon_xx[:-1, :, :]
        epsilon_xy = epsilon_xy[1:, :, :]
        epsilon_xz = epsilon_xz[1:, :, :]
        epsilon_yx = epsilon_yx[1:, :, :]
        epsilon_yy = epsilon_yy[1:, :, :]
        epsilon_yz = epsilon_yz[1:, :, :]
        epsilon_zx = epsilon_zx[1:, :, :]
        epsilon_zy = epsilon_zy[1:, :, :]
        epsilon_zz = epsilon_zz[1:, :, :]

    if yBC == "PEC":
        epsilon_xx = epsilon_xx[:, 1:, :]
        epsilon_xy = epsilon_xy[:, 1:, :]
        epsilon_xz = epsilon_xz[:, 1:, :]
        epsilon_yx = epsilon_yx[:, 1:, :]
        epsilon_yz = epsilon_yz[:, 1:, :]
        epsilon_zx = epsilon_zx[:, 1:, :]
        epsilon_zy = epsilon_zy[:, 1:, :]
        epsilon_zz = epsilon_zz[:, 1:, :]
    elif yBC == "PMC":
        epsilon_yy = epsilon_yy[:, :-1, :]
    elif yBC == "PECPMC":
        epsilon_xx = epsilon_xx[:, 1:, :]
        epsilon_xy = epsilon_xy[:, 1:, :]
        epsilon_xz = epsilon_xz[:, 1:, :]
        epsilon_yx = epsilon_yx[:, 1:, :]
        epsilon_yy = epsilon_yy[:, :-1, :]
        epsilon_yz = epsilon_yz[:, 1:, :]
        epsilon_zx = epsilon_zx[:, 1:, :]
        epsilon_zy = epsilon_zy[:, 1:, :]
        epsilon_zz = epsilon_zz[:, 1:, :]

    if zBC == "PEC":
        epsilon_xx = epsilon_xx[:, :, 1:]
        epsilon_xy = epsilon_xy[:, :, 1:]
        epsilon_xz = epsilon_xz[:, :, 1:]
        epsilon_yx = epsilon_yx[:, :, 1:]
        epsilon_yy = epsilon_yy[:, :, 1:]
        epsilon_yz = epsilon_yz[:, :, 1:]
        epsilon_zx = epsilon_zx[:, :, 1:]
        epsilon_zy = epsilon_zy[:, :, 1:]
    elif zBC == "PMC":
        epsilon_zz = epsilon_zz[:, :, :-1]
    elif zBC == "PECPMC":
        epsilon_xx = epsilon_xx[:, :, 1:]
        epsilon_xy = epsilon_xy[:, :, 1:]
        epsilon_xz = epsilon_xz[:, :, 1:]
        epsilon_yx = epsilon_yx[:, :, 1:]
        epsilon_yy = epsilon_yy[:, :, 1:]
        epsilon_yz = epsilon_yz[:, :, 1:]
        epsilon_zx = epsilon_zx[:, :, 1:]
        epsilon_zy = epsilon_zy[:, :, 1:]
        epsilon_zz = epsilon_zz[:, :, :-1]

    return (
        epsilon_xx,
        epsilon_xy,
        epsilon_xz,
        epsilon_yx,
        epsilon_yy,
        epsilon_yz,
        epsilon_zx,
        epsilon_zy,
        epsilon_zz,
    )


def _nearest_cuboid_face(point: np.ndarray, cuboid: Cuboid) -> tuple[int, str, float]:
    lower = cuboid.lower
    upper = cuboid.upper
    outside_candidates: list[tuple[float, int, str, float]] = []
    for axis in range(cuboid.ndim):
        coord = point[axis]
        if coord < lower[axis]:
            outside_candidates.append((lower[axis] - coord, axis, "lower", lower[axis]))
        elif coord > upper[axis]:
            outside_candidates.append((coord - upper[axis], axis, "upper", upper[axis]))
    if outside_candidates:
        _, axis, side, coordinate = min(outside_candidates, key=lambda item: item[0])
        return axis, side, coordinate

    candidates: list[tuple[float, int, str, float]] = []
    for axis in range(cuboid.ndim):
        coord = point[axis]
        candidates.append((coord - lower[axis], axis, "lower", lower[axis]))
        candidates.append((upper[axis] - coord, axis, "upper", upper[axis]))
    _, axis, side, coordinate = min(candidates, key=lambda item: item[0])
    return axis, side, coordinate


def _cuboid_surface_point_and_normal(point: np.ndarray, cuboid: Cuboid) -> tuple[np.ndarray, np.ndarray]:
    point_arr = np.asarray(point, dtype=float)
    lower = cuboid.lower
    upper = cuboid.upper
    clamped = np.clip(point_arr, lower, upper)
    outside_vector = point_arr - clamped
    outside_distance = np.linalg.norm(outside_vector)
    if outside_distance > 1e-15:
        return clamped, outside_vector / outside_distance

    axis, side, coordinate = _nearest_cuboid_face(point_arr, cuboid)
    surface_point = point_arr.copy()
    surface_point[axis] = coordinate
    direction = coordinate - point_arr[axis]
    if math.isclose(direction, 0.0, rel_tol=0.0, abs_tol=1e-15):
        direction = -1.0 if side == "lower" else 1.0
    normal = np.zeros(cuboid.ndim, dtype=float)
    normal[axis] = math.copysign(1.0, direction)
    return surface_point, normal


def _ball_surface_point_and_normal(point: np.ndarray, ball: Ball) -> tuple[np.ndarray, np.ndarray]:
    point_arr = np.asarray(point, dtype=float)
    center = np.asarray(ball.center, dtype=float)
    normal = point_arr - center
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-15:
        normal = np.zeros(ball.ndim, dtype=float)
        normal[0] = 1.0
    else:
        normal = normal / normal_norm
    surface_point = center + ball.radius * normal
    return surface_point, normal


def _shape_surface_point_and_normal(point: np.ndarray, shape: Cuboid | Ball) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(shape, Cuboid):
        return _cuboid_surface_point_and_normal(point, shape)
    if isinstance(shape, Ball):
        return _ball_surface_point_and_normal(point, shape)
    raise NotImplementedError("Only Cuboid and Ball subpixel smoothing objects are supported.")


def _corner_bits(vxl_low: np.ndarray, vxl_high: np.ndarray, normal: np.ndarray, nr0: float) -> tuple[int, int]:
    cbits = 0
    bit = 1
    n_on = 0
    for z_side in (0, 1):
        for y_side in (0, 1):
            for x_side in (0, 1):
                corner = np.array(
                    [
                        vxl_high[0] if x_side else vxl_low[0],
                        vxl_high[1] if y_side else vxl_low[1],
                        vxl_high[2] if z_side else vxl_low[2],
                    ],
                    dtype=float,
                )
                nr = float(np.dot(normal, corner))
                if nr <= nr0:
                    cbits |= bit
                    if nr == nr0:
                        n_on += 1
                bit <<= 1
    return cbits, n_on


def _is_quadsect(cbits: int) -> bool:
    return cbits in {0x0F, 0xF0, 0x33, 0xCC, 0x55, 0xAA}


def _edge_dir_quadsect(cbits: int) -> int:
    if cbits in {0x0F, 0xF0}:
        return 2
    if cbits in {0x33, 0xCC}:
        return 1
    return 0


def _relative_volume_quadsect(vxl_low: np.ndarray, vxl_high: np.ndarray, normal: np.ndarray, nr0: float, cbits: int) -> float:
    w_axis = _edge_dir_quadsect(cbits)
    width = vxl_high[w_axis] - vxl_low[w_axis]
    uvw = ((1, 2, 0), (2, 0, 1), (0, 1, 2))
    u_axis, v_axis, _ = uvw[w_axis]
    nu = normal[u_axis]
    nv = normal[v_axis]
    nw = normal[w_axis]
    mean_intercepts = 4 * nr0
    for v_side in (0, 1):
        for u_side in (0, 1):
            u_coord = vxl_high[u_axis] if u_side else vxl_low[u_axis]
            v_coord = vxl_high[v_axis] if v_side else vxl_low[v_axis]
            mean_intercepts -= nu * u_coord + nv * v_coord
    mean_intercepts /= nw * 4 * width
    side_coord = vxl_low[w_axis] if nw > 0 else vxl_high[w_axis]
    return float(abs(mean_intercepts - side_coord / width))


def _relative_volume_gensect(vxl_low: np.ndarray, vxl_high: np.ndarray, normal: np.ndarray, nr0: float) -> float:
    corner = np.where(normal < 0, vxl_high, vxl_low)
    widths = vxl_high - vxl_low
    normal_corner = normal * corner
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.abs((((nr0 - np.sum(normal_corner)) + normal_corner) / normal - corner) / widths)
    rmax, rmid, rmin = sorted([float(value) for value in ratios], reverse=True)

    tmax = 1 - 1 / rmax
    relative_core = 1 + tmax + tmax**2
    if math.isfinite(rmax):
        if rmid > 1:
            tmid = 1 - 1 / rmid
            relative_core -= rmax * tmid**3
        if rmin > 1:
            tmin = 1 - 1 / rmin
            relative_core -= rmax * tmin**3
    return float(relative_core * rmin * rmid / 6)


def _plane_volume_fraction_3d(vxl_low: np.ndarray, vxl_high: np.ndarray, normal: np.ndarray, surface_point: np.ndarray) -> float:
    nr0 = float(np.dot(normal, surface_point))
    cbits, n_on = _corner_bits(vxl_low, vxl_high, normal, nr0)
    n_in = int(cbits.bit_count())

    if n_in == 8:
        volume = 1.0
    elif n_in - n_on == 0:
        volume = 0.0
    elif _is_quadsect(cbits):
        volume = _relative_volume_quadsect(vxl_low, vxl_high, normal, nr0, cbits)
    elif n_in <= 4:
        volume = _relative_volume_gensect(vxl_low, vxl_high, normal, nr0)
    else:
        volume = 1.0 - _relative_volume_gensect(vxl_low, vxl_high, -normal, -nr0)
    return float(np.clip(volume, 0.0, 1.0))


def _local_planar_fill_fraction(point: np.ndarray, delta_x: float, shape: Cuboid | Ball) -> float:
    surface_point, normal = _shape_surface_point_and_normal(point, shape)
    low = np.asarray(point, dtype=float) - delta_x / 2
    high = np.asarray(point, dtype=float) + delta_x / 2
    if shape.ndim == 2:
        low_3d = np.array([low[0], low[1], 0.0], dtype=float)
        high_3d = np.array([high[0], high[1], 1.0], dtype=float)
        normal_3d = np.array([normal[0], normal[1], 0.0], dtype=float)
        surface_3d = np.array([surface_point[0], surface_point[1], 0.0], dtype=float)
        return _plane_volume_fraction_3d(low_3d, high_3d, normal_3d, surface_3d)
    return _plane_volume_fraction_3d(low, high, normal, surface_point)


def _corner_offsets(delta_x: float, ndim: int) -> np.ndarray:
    return np.array(list(product((-delta_x / 2, delta_x / 2), repeat=ndim)), dtype=float)


def _corners_inside_cuboid(point: np.ndarray, delta_x: float, shape: Cuboid | Ball) -> np.ndarray:
    return np.array([shape.contains(point + offset) for offset in _corner_offsets(delta_x, shape.ndim)], dtype=bool)


def _shape_surface_normal(point: np.ndarray, shape: Cuboid | Ball, vector_ndim: int, axis_offset: int = 0) -> np.ndarray:
    normal = np.zeros(vector_ndim, dtype=float)
    _, shape_normal = _shape_surface_point_and_normal(point, shape)
    normal[axis_offset : axis_offset + shape.ndim] = shape_normal
    return normal


def _orthonormal_basis_from_normal(normal: np.ndarray) -> np.ndarray:
    n0 = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(n0)
    if norm == 0:
        raise ValueError("surface normal must be nonzero.")
    n0 = n0 / norm
    helper = np.eye(3)[int(np.argmin(np.abs(np.eye(3) @ n0)))]
    tangent_1 = helper - n0 * float(np.dot(n0, helper))
    tangent_1 = tangent_1 / np.linalg.norm(tangent_1)
    tangent_2 = np.cross(n0, tangent_1)
    return np.column_stack((n0, tangent_1, tangent_2))


def _tau_trans(epsilon: np.ndarray) -> np.ndarray:
    epsilon_11 = epsilon[0, 0]
    epsilon_21 = epsilon[1, 0]
    epsilon_31 = epsilon[2, 0]
    epsilon_12 = epsilon[0, 1]
    epsilon_22 = epsilon[1, 1]
    epsilon_32 = epsilon[2, 1]
    epsilon_13 = epsilon[0, 2]
    epsilon_23 = epsilon[1, 2]
    epsilon_33 = epsilon[2, 2]
    return np.array(
        [
            [-1 / epsilon_11, epsilon_12 / epsilon_11, epsilon_13 / epsilon_11],
            [epsilon_21 / epsilon_11, epsilon_22 - epsilon_21 * epsilon_12 / epsilon_11, epsilon_23 - epsilon_21 * epsilon_13 / epsilon_11],
            [epsilon_31 / epsilon_11, epsilon_32 - epsilon_31 * epsilon_12 / epsilon_11, epsilon_33 - epsilon_31 * epsilon_13 / epsilon_11],
        ],
        dtype=np.result_type(epsilon, np.float64),
    )


def _tau_inverse_trans(tau: np.ndarray) -> np.ndarray:
    tau_11 = tau[0, 0]
    tau_21 = tau[1, 0]
    tau_31 = tau[2, 0]
    tau_12 = tau[0, 1]
    tau_22 = tau[1, 1]
    tau_32 = tau[2, 1]
    tau_13 = tau[0, 2]
    tau_23 = tau[1, 2]
    tau_33 = tau[2, 2]
    return np.array(
        [
            [-1 / tau_11, -tau_12 / tau_11, -tau_13 / tau_11],
            [-tau_21 / tau_11, tau_22 - tau_21 * tau_12 / tau_11, tau_23 - tau_21 * tau_13 / tau_11],
            [-tau_31 / tau_11, tau_32 - tau_31 * tau_12 / tau_11, tau_33 - tau_31 * tau_13 / tau_11],
        ],
        dtype=np.result_type(tau, np.float64),
    )


def _kottke_smoothing(
    vol_frac: float,
    normal: np.ndarray,
    epsilon_object: np.ndarray,
    epsilon_voxel: np.ndarray,
) -> np.ndarray:
    basis = _orthonormal_basis_from_normal(normal)
    tau_voxel = _tau_trans(basis.T @ epsilon_voxel @ basis)
    tau_object = _tau_trans(basis.T @ epsilon_object @ basis)
    tau_avg = tau_object * vol_frac + tau_voxel * (1 - vol_frac)
    return basis @ _tau_inverse_trans(tau_avg) @ basis.T


def _smooth_2d_tm_cuboids(
    delta_x: float,
    domain: Cuboid,
    domain_epsilon: complex,
    object_list: Sequence[Cuboid],
    object_epsilon_list: Sequence[complex],
    yBC: str,
    zBC: str,
    without_sb: bool,
) -> np.ndarray:
    domain, objects = _normalized_domain_and_objects(delta_x, domain, object_list)
    objects, epsilons = _add_periodic_images(domain, objects, list(object_epsilon_list), yBC, zBC)
    counts = np.rint((domain.upper - domain.lower) / delta_x).astype(int)
    if np.any(counts <= 0):
        raise ValueError("Smoothing domain must contain at least one grid site in each direction.")

    dtype = np.result_type(domain_epsilon, *epsilons, np.float64)
    epsilon_xx = np.full(tuple(counts.tolist()), domain_epsilon, dtype=dtype)
    y_coords = domain.lower[0] + np.arange(counts[0], dtype=float) * delta_x
    z_coords = domain.lower[1] + np.arange(counts[1], dtype=float) * delta_x

    for obj, eps_value in zip(objects, epsilons):
        for jj, y_coord in enumerate(y_coords):
            for kk, z_coord in enumerate(z_coords):
                point = np.array([y_coord, z_coord], dtype=float)
                if without_sb:
                    if obj.contains(point):
                        epsilon_xx[jj, kk] = eps_value
                    continue
                corners_inside = _corners_inside_cuboid(point, delta_x, obj)
                if bool(np.all(corners_inside)):
                    epsilon_xx[jj, kk] = eps_value
                elif not bool(np.all(~corners_inside)):
                    # For isotropic 2D TM, Julia's Kottke tensor smoothing
                    # reduces the x-directed permittivity to the arithmetic
                    # mix over the local planar fill fraction.
                    vol_frac = _local_planar_fill_fraction(point, delta_x, obj)
                    epsilon_xx[jj, kk] = eps_value * vol_frac + epsilon_xx[jj, kk] * (1 - vol_frac)
    return _pick_epsilon_2d_tm(epsilon_xx, yBC, zBC)


def _smooth_inverse_epsilon_site(
    inv_epsilon_site: np.ndarray,
    point: np.ndarray,
    delta_x: float,
    obj: Cuboid,
    eps_value: complex,
    without_sb: bool,
) -> np.ndarray:
    if without_sb:
        if obj.contains(point):
            return np.eye(3, dtype=inv_epsilon_site.dtype) / eps_value
        return inv_epsilon_site

    corners_inside = _corners_inside_cuboid(point, delta_x, obj)
    if bool(np.all(corners_inside)):
        return np.eye(3, dtype=inv_epsilon_site.dtype) / eps_value
    if bool(np.all(~corners_inside)):
        return inv_epsilon_site

    vol_frac = _local_planar_fill_fraction(point, delta_x, obj)
    normal = _shape_surface_normal(point, obj, vector_ndim=3, axis_offset=1)
    epsilon_object = np.eye(3, dtype=inv_epsilon_site.dtype) * eps_value
    epsilon_voxel = np.linalg.inv(inv_epsilon_site)
    return np.linalg.inv(_kottke_smoothing(vol_frac, normal, epsilon_object, epsilon_voxel))


def _smooth_2d_te_cuboids(
    delta_x: float,
    domain: Cuboid,
    domain_epsilon: complex,
    object_list: Sequence[Cuboid],
    object_epsilon_list: Sequence[complex],
    yBC: str,
    zBC: str,
    without_sb: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    domain, objects = _normalized_domain_and_objects(delta_x, domain, object_list)
    objects, epsilons = _add_periodic_images(domain, objects, list(object_epsilon_list), yBC, zBC)
    counts = np.rint((domain.upper - domain.lower) / delta_x).astype(int)
    if np.any(counts <= 0):
        raise ValueError("Smoothing domain must contain at least one grid site in each direction.")

    dtype = np.result_type(domain_epsilon, *epsilons, np.float64)
    shape = (*tuple(counts.tolist()), 3, 3)
    inv_domain_epsilon = 1 / domain_epsilon
    inv_epsilon_Eo_site = np.empty(shape, dtype=dtype)
    inv_epsilon_Ey_site = np.empty(shape, dtype=dtype)
    inv_epsilon_Ez_site = np.empty(shape, dtype=dtype)
    identity = np.eye(3, dtype=dtype) * inv_domain_epsilon
    inv_epsilon_Eo_site[...] = identity
    inv_epsilon_Ey_site[...] = identity
    inv_epsilon_Ez_site[...] = identity

    y_lower, z_lower = domain.lower
    ny, nz = counts.tolist()
    ez_y_coords = y_lower + np.arange(ny, dtype=float) * delta_x
    ez_z_coords = z_lower + (np.arange(nz, dtype=float) + 0.5) * delta_x
    ey_y_coords = y_lower + (np.arange(ny, dtype=float) + 0.5) * delta_x
    ey_z_coords = z_lower + np.arange(nz, dtype=float) * delta_x

    for obj, eps_value in zip(objects, epsilons):
        for jj in range(ny):
            for kk in range(nz):
                eo_point = np.array([ez_y_coords[jj], ey_z_coords[kk]], dtype=float)
                ey_point = np.array([ey_y_coords[jj], ey_z_coords[kk]], dtype=float)
                ez_point = np.array([ez_y_coords[jj], ez_z_coords[kk]], dtype=float)
                inv_epsilon_Eo_site[jj, kk] = _smooth_inverse_epsilon_site(
                    inv_epsilon_Eo_site[jj, kk], eo_point, delta_x, obj, eps_value, without_sb
                )
                inv_epsilon_Ey_site[jj, kk] = _smooth_inverse_epsilon_site(
                    inv_epsilon_Ey_site[jj, kk], ey_point, delta_x, obj, eps_value, without_sb
                )
                inv_epsilon_Ez_site[jj, kk] = _smooth_inverse_epsilon_site(
                    inv_epsilon_Ez_site[jj, kk], ez_point, delta_x, obj, eps_value, without_sb
                )

    return _pick_inv_epsilon_2d_te(inv_epsilon_Ey_site, inv_epsilon_Ez_site, inv_epsilon_Eo_site, yBC, zBC)


def _smooth_epsilon_site(
    epsilon_site: np.ndarray,
    point: np.ndarray,
    delta_x: float,
    obj: Cuboid,
    eps_value: complex,
    without_sb: bool,
) -> np.ndarray:
    if without_sb:
        if obj.contains(point):
            return np.eye(3, dtype=epsilon_site.dtype) * eps_value
        return epsilon_site

    corners_inside = _corners_inside_cuboid(point, delta_x, obj)
    if bool(np.all(corners_inside)):
        return np.eye(3, dtype=epsilon_site.dtype) * eps_value
    if bool(np.all(~corners_inside)):
        return epsilon_site

    vol_frac = _local_planar_fill_fraction(point, delta_x, obj)
    normal = _shape_surface_normal(point, obj, vector_ndim=3)
    epsilon_object = np.eye(3, dtype=epsilon_site.dtype) * eps_value
    return _kottke_smoothing(vol_frac, normal, epsilon_object, epsilon_site)


def _smooth_3d_cuboids(
    delta_x: float,
    domain: Cuboid,
    domain_epsilon: complex,
    object_list: Sequence[Cuboid],
    object_epsilon_list: Sequence[complex],
    xBC: str,
    yBC: str,
    zBC: str,
    without_sb: bool,
) -> tuple[np.ndarray, ...]:
    domain, objects = _normalized_domain_and_objects(delta_x, domain, object_list)
    objects, epsilons = _add_periodic_images(domain, objects, list(object_epsilon_list), xBC, yBC, zBC)
    counts = np.rint((domain.upper - domain.lower) / delta_x).astype(int)
    if np.any(counts <= 0):
        raise ValueError("Smoothing domain must contain at least one grid site in each direction.")

    dtype = np.result_type(domain_epsilon, *epsilons, np.float64)
    shape = (*tuple(counts.tolist()), 3, 3)
    identity = np.eye(3, dtype=dtype) * domain_epsilon
    epsilon_Eo_site = np.empty(shape, dtype=dtype)
    epsilon_Ex_site = np.empty(shape, dtype=dtype)
    epsilon_Ey_site = np.empty(shape, dtype=dtype)
    epsilon_Ez_site = np.empty(shape, dtype=dtype)
    epsilon_Eo_site[...] = identity
    epsilon_Ex_site[...] = identity
    epsilon_Ey_site[...] = identity
    epsilon_Ez_site[...] = identity

    x_lower, y_lower, z_lower = domain.lower
    nx, ny, nz = counts.tolist()
    eo_x_coords = x_lower + np.arange(nx, dtype=float) * delta_x
    eo_y_coords = y_lower + np.arange(ny, dtype=float) * delta_x
    eo_z_coords = z_lower + np.arange(nz, dtype=float) * delta_x
    ex_x_coords = x_lower + (np.arange(nx, dtype=float) + 0.5) * delta_x
    ex_y_coords = eo_y_coords
    ex_z_coords = eo_z_coords
    ey_x_coords = eo_x_coords
    ey_y_coords = y_lower + (np.arange(ny, dtype=float) + 0.5) * delta_x
    ey_z_coords = eo_z_coords
    ez_x_coords = eo_x_coords
    ez_y_coords = eo_y_coords
    ez_z_coords = z_lower + (np.arange(nz, dtype=float) + 0.5) * delta_x

    for obj, eps_value in zip(objects, epsilons):
        for ii in range(nx):
            for jj in range(ny):
                for kk in range(nz):
                    eo_point = np.array([eo_x_coords[ii], eo_y_coords[jj], eo_z_coords[kk]], dtype=float)
                    ex_point = np.array([ex_x_coords[ii], ex_y_coords[jj], ex_z_coords[kk]], dtype=float)
                    ey_point = np.array([ey_x_coords[ii], ey_y_coords[jj], ey_z_coords[kk]], dtype=float)
                    ez_point = np.array([ez_x_coords[ii], ez_y_coords[jj], ez_z_coords[kk]], dtype=float)
                    epsilon_Eo_site[ii, jj, kk] = _smooth_epsilon_site(
                        epsilon_Eo_site[ii, jj, kk], eo_point, delta_x, obj, eps_value, without_sb
                    )
                    epsilon_Ex_site[ii, jj, kk] = _smooth_epsilon_site(
                        epsilon_Ex_site[ii, jj, kk], ex_point, delta_x, obj, eps_value, without_sb
                    )
                    epsilon_Ey_site[ii, jj, kk] = _smooth_epsilon_site(
                        epsilon_Ey_site[ii, jj, kk], ey_point, delta_x, obj, eps_value, without_sb
                    )
                    epsilon_Ez_site[ii, jj, kk] = _smooth_epsilon_site(
                        epsilon_Ez_site[ii, jj, kk], ez_point, delta_x, obj, eps_value, without_sb
                    )

    return _pick_epsilon_3d(epsilon_Eo_site, epsilon_Ex_site, epsilon_Ey_site, epsilon_Ez_site, xBC, yBC, zBC)


def mesti_subpixel_smoothing(
    delta_x: float,
    domain: Cuboid,
    domain_epsilon: complex,
    object_list: Sequence[Cuboid],
    object_epsilon_list: Sequence[complex],
    yBC: str,
    zBC: str,
    use_2D_TM: bool = True,
    use_2D_TE: bool = False,
    without_sb: bool = False,
) -> np.ndarray | tuple[np.ndarray, ...] | tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Build 2D TM/TE or 3D subpixel profiles from Cuboid/Ball objects.

    This is a fixture-backed first slice of Julia ``mesti_subpixel_smoothing``.
    It supports 2D ``Cuboid`` domains and objects, isotropic scalar
    permittivities, the TM ``epsilon_xx`` output, and TE inverse-epsilon
    ``(yy, zz, yz)`` outputs. It also supports the 3D tensor-output overload
    with scalar isotropic permittivities. Curved ``Ball`` objects use the same
    local-planar Kottke smoothing path as Julia's ``GeometryPrimitives``
    implementation.
    """

    if not isinstance(domain, Cuboid):
        raise TypeError("domain must be a Cuboid instance.")
    if delta_x <= 0:
        raise ValueError("delta_x must be positive.")
    if len(object_list) != len(object_epsilon_list):
        raise ValueError("object_list and object_epsilon_list must have the same length.")
    domain_eps = np.asarray(domain_epsilon).reshape(-1)
    if domain_eps.size != 1:
        raise ValueError("domain_epsilon must be a scalar.")
    object_eps = [np.asarray(value).reshape(-1) for value in object_epsilon_list]
    if any(value.size != 1 for value in object_eps):
        raise ValueError("object_epsilon_list entries must be scalar.")

    delta_x_float = float(delta_x)
    domain_epsilon_scalar = domain_eps[0].item()
    object_epsilon_scalars = [value[0].item() for value in object_eps]

    if domain.ndim == 3:
        if not isinstance(use_2D_TM, str):
            raise TypeError("3D subpixel smoothing must be called as (..., xBC, yBC, zBC, without_sb=False).")
        if not isinstance(use_2D_TE, (bool, np.bool_)):
            raise TypeError("3D subpixel smoothing expects the optional ninth positional argument to be without_sb.")
        _validate_smoothing_objects(object_list, 3)
        xBC_normalized = _convert_bc_sbpsm(yBC, "x")
        yBC_normalized = _convert_bc_sbpsm(zBC, "y")
        zBC_normalized = _convert_bc_sbpsm(use_2D_TM, "z")
        without_sb_3d = bool(without_sb) or bool(use_2D_TE)
        return _smooth_3d_cuboids(
            delta_x_float,
            domain,
            domain_epsilon_scalar,
            object_list,
            object_epsilon_scalars,
            xBC_normalized,
            yBC_normalized,
            zBC_normalized,
            without_sb_3d,
        )

    if domain.ndim != 2:
        raise NotImplementedError("Only 2D or 3D Cuboid subpixel smoothing is supported.")
    if not isinstance(use_2D_TM, (bool, np.bool_)) or not isinstance(use_2D_TE, (bool, np.bool_)):
        raise TypeError("2D subpixel smoothing expects boolean use_2D_TM and use_2D_TE flags.")
    if not use_2D_TM and not use_2D_TE:
        raise ValueError("In 2D case, use_2D_TM and/or use_2D_TE should be true.")
    _validate_smoothing_objects(object_list, 2)

    yBC_normalized = _convert_bc_sbpsm(yBC, "y")
    zBC_normalized = _convert_bc_sbpsm(zBC, "z")
    epsilon_xx = None
    inv_epsilon = None
    if use_2D_TM:
        epsilon_xx = _smooth_2d_tm_cuboids(
            delta_x_float,
            domain,
            domain_epsilon_scalar,
            object_list,
            object_epsilon_scalars,
            yBC_normalized,
            zBC_normalized,
            bool(without_sb),
        )
    if use_2D_TE:
        inv_epsilon = _smooth_2d_te_cuboids(
            delta_x_float,
            domain,
            domain_epsilon_scalar,
            object_list,
            object_epsilon_scalars,
            yBC_normalized,
            zBC_normalized,
            bool(without_sb),
        )
    if use_2D_TM and use_2D_TE:
        return epsilon_xx, inv_epsilon
    if use_2D_TE:
        return inv_epsilon
    return epsilon_xx
