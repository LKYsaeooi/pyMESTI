"""Boundary-condition and PML helpers translated from MESTI.jl."""

from __future__ import annotations

import numbers
from typing import Sequence

import numpy as np
from scipy import sparse

from .types import PML


def _is_number(value: object) -> bool:
    return isinstance(value, numbers.Number)


def _one_to(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    return np.arange(1, n + 1, dtype=float)


def _reverse_one_to(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    return np.arange(n, 0, -1, dtype=int)


def _diag_matrix(rows: int, cols: int, specs: Sequence[tuple[int, np.ndarray]]) -> sparse.csc_matrix:
    result = sparse.csc_matrix((rows, cols), dtype=np.complex128)
    for offset, values in specs:
        arr = np.asarray(values, dtype=np.complex128)
        if arr.size:
            result = result + sparse.diags(
                [arr],
                [offset],
                shape=(rows, cols),
                dtype=np.complex128,
                format="csc",
            )
    return result


def convert_BC(BC: str | numbers.Number, direction: str) -> str | numbers.Number:
    """Normalize MESTI boundary condition labels."""

    if _is_number(BC):
        return BC

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
        raise ValueError(
            "To use Bloch periodic boundary condition in "
            f"{direction}-direction, set {direction}BC to k{direction}_B*p_{direction}."
        )
    raise ValueError(f'Input argument {direction}BC = "{BC}" is not a supported option.')


def convert_BC_to_transverse(
    BC: str | numbers.Number,
    component: str,
    direction: str,
) -> str | numbers.Number:
    """Convert PEC/PMC-style BC labels to transverse mode BC labels."""

    if _is_number(BC):
        return BC
    if BC == "PEC":
        return "Neumann" if direction == component else "Dirichlet"
    if BC == "PMC":
        return "Dirichlet" if direction == component else "Neumann"
    if BC == "PECPMC":
        return "NeumannDirichlet" if direction == component else "DirichletNeumann"
    if BC == "PMCPEC":
        return "DirichletNeumann" if direction == component else "NeumannDirichlet"
    if BC == "periodic":
        return "periodic"
    if str(BC).lower() == "bloch":
        raise ValueError(
            "To use Bloch periodic boundary condition in "
            f"{direction}-direction, set {direction}BC to k{direction}_B*p_{direction}."
        )
    raise ValueError(f'Input argument {direction}BC = "{BC}" is not a supported option.')


def mesti_set_PML_params(
    pml: Sequence[PML],
    k0dx: float | complex,
    epsilon_bg: Sequence[float],
    direction: str,
) -> list[PML]:
    """Fill default PML parameters following ``mesti_set_PML_params.jl``."""

    pml = list(pml)
    if len(pml) != 2:
        raise ValueError(f"{direction}PML must contain two PML objects.")

    if pml[0].npixels == 0 and pml[1].npixels == 0:
        return pml

    if len(epsilon_bg) != 2:
        raise ValueError("epsilon_bg must contain one background value per PML side.")

    for idx, layer in enumerate(pml):
        if layer.npixels is None:
            raise ValueError(f"{direction}PML[{idx}].npixels must be set.")
        if layer.npixels < 0:
            raise ValueError(f"{direction}PML[{idx}].npixels must be non-negative.")
        if layer.npixels == 0:
            continue

        wavelength_over_dx = ((2 * np.pi) / k0dx) / np.sqrt(epsilon_bg[idx])

        if layer.power_sigma is None:
            layer.power_sigma = 3
        elif layer.power_sigma < 0:
            raise ValueError(f"{direction}PML[{idx}].power_sigma must be non-negative.")

        if layer.sigma_max_over_omega is None:
            layer.sigma_max_over_omega = float(-3.0138 + 0.9303 * wavelength_over_dx**1.0128)
        elif layer.sigma_max_over_omega < 0:
            raise ValueError(f"{direction}PML[{idx}].sigma_max_over_omega must be non-negative.")

        if layer.kappa_max is None:
            layer.kappa_max = float(-2.0944 + 0.6617 * wavelength_over_dx**1.0467)
        elif layer.kappa_max < 1:
            raise ValueError(f"{direction}PML[{idx}].kappa_max must be at least 1.")

        if layer.power_kappa is None:
            layer.power_kappa = 3
        elif layer.power_kappa < 0:
            raise ValueError(f"{direction}PML[{idx}].power_kappa must be non-negative.")

        if layer.alpha_max_over_omega is None:
            layer.alpha_max_over_omega = 0
        elif layer.alpha_max_over_omega < 0:
            raise ValueError(f"{direction}PML[{idx}].alpha_max_over_omega must be non-negative.")

        if layer.power_alpha is None:
            layer.power_alpha = 1
        elif layer.power_alpha < 0:
            raise ValueError(f"{direction}PML[{idx}].power_alpha must be non-negative.")

    return pml


def _build_e_matrices(n_E: int, BC: str | numbers.Number, direction: str) -> tuple[sparse.csc_matrix, sparse.csc_matrix, str, complex]:
    if _is_number(BC):
        kLambda = complex(BC)
        BC = "Bloch"
    elif BC == "periodic":
        kLambda = 0j
        BC = "Bloch"
    else:
        kLambda = 0j

    one = np.ones(n_E, dtype=np.complex128)
    if BC == "Bloch":
        phase = np.exp(1j * kLambda)
        ddx = _diag_matrix(
            n_E,
            n_E,
            [
                (1, np.ones(n_E - 1)),
                (0, -one),
                (1 - n_E, np.array([phase])),
            ],
        )
        avg = _diag_matrix(
            n_E,
            n_E,
            [
                (1, np.ones(n_E - 1) / 2),
                (0, one / 2),
                (1 - n_E, np.array([phase / 2])),
            ],
        )
    elif BC == "PEC":
        ddx = _diag_matrix(n_E + 1, n_E, [(0, one), (-1, -one)])
        avg = _diag_matrix(n_E + 1, n_E, [(0, one / 2), (-1, one / 2)])
    elif BC == "PMC":
        ddx = _diag_matrix(n_E - 1, n_E, [(1, np.ones(n_E - 1)), (0, -np.ones(n_E - 1))])
        avg = _diag_matrix(n_E - 1, n_E, [(1, np.ones(n_E - 1) / 2), (0, np.ones(n_E - 1) / 2)])
    elif BC == "PECPMC":
        ddx = _diag_matrix(n_E, n_E, [(0, one), (-1, -np.ones(n_E - 1))])
        avg = _diag_matrix(n_E, n_E, [(0, one / 2), (-1, np.ones(n_E - 1) / 2)])
    elif BC == "PMCPEC":
        ddx = _diag_matrix(n_E, n_E, [(1, np.ones(n_E - 1)), (0, -one)])
        avg = _diag_matrix(n_E, n_E, [(1, np.ones(n_E - 1) / 2), (0, one / 2)])
    else:
        raise ValueError(f'Input argument {direction}BC = "{BC}" is not a supported option.')
    return ddx, avg, str(BC), kLambda


def build_ddx_E(
    n_E: int,
    BC: str | numbers.Number,
    pml: Sequence[PML],
    direction: str,
) -> tuple[sparse.csc_matrix, sparse.csc_matrix, np.ndarray, np.ndarray, list[np.ndarray | None]]:
    """Build first-derivative and averaging matrices acting on E-grid fields."""

    ddx, avg, normalized_BC, _ = _build_e_matrices(n_E, BC, direction)
    n_H = ddx.shape[0]
    s_E = np.ones(n_E, dtype=np.complex128)
    s_H = np.ones(n_H, dtype=np.complex128)

    pml = list(pml)
    if len(pml) != 2:
        raise ValueError(f"{direction}PML must contain two PML objects.")
    if pml[0].npixels == 0 and pml[1].npixels == 0:
        return ddx, avg, s_E, s_H, [None, None]

    npixels = [int(pml[0].npixels or 0), int(pml[1].npixels or 0)]
    if sum(npixels) >= n_E:
        raise ValueError(
            f"Total pixels in {direction} direction must exceed PML pixels "
            f"{npixels[0]} + {npixels[1]}."
        )

    n = n_E
    if normalized_BC == "Bloch":
        npixels_effective = [npixels[0] + 0.5, npixels[1] + 0.5]
        p_PML_2 = [
            (_one_to(npixels[0]) - 0.5) / npixels_effective[0],
            (_one_to(npixels[1] + 1) - 0.5) / npixels_effective[1],
        ]
        ind_PML_2 = [
            _reverse_one_to(npixels[0]),
            (n + 1) - _reverse_one_to(npixels[1] + 1),
        ]
    elif normalized_BC == "PEC":
        npixels_effective = [npixels[0] + 1, npixels[1] + 1]
        p_PML_2 = [
            (_one_to(npixels[0] + 1) - 0.5) / npixels_effective[0],
            (_one_to(npixels[1] + 1) - 0.5) / npixels_effective[1],
        ]
        ind_PML_2 = [
            _reverse_one_to(npixels[0] + 1),
            (n + 2) - _reverse_one_to(npixels[1] + 1),
        ]
    elif normalized_BC == "PMC":
        npixels_effective = [npixels[0] + 0.5, npixels[1] + 0.5]
        p_PML_2 = [
            (_one_to(npixels[0]) - 0.5) / npixels_effective[0],
            (_one_to(npixels[1]) - 0.5) / npixels_effective[1],
        ]
        ind_PML_2 = [
            _reverse_one_to(npixels[0]),
            n - _reverse_one_to(npixels[1]),
        ]
    elif normalized_BC == "PECPMC":
        npixels_effective = [npixels[0] + 1, npixels[1] + 0.5]
        p_PML_2 = [
            (_one_to(npixels[0] + 1) - 0.5) / npixels_effective[0],
            (_one_to(npixels[1]) - 0.5) / npixels_effective[1],
        ]
        ind_PML_2 = [
            _reverse_one_to(npixels[0] + 1),
            (n + 1) - _reverse_one_to(npixels[1]),
        ]
    elif normalized_BC == "PMCPEC":
        npixels_effective = [npixels[0] + 0.5, npixels[1] + 1]
        p_PML_2 = [
            (_one_to(npixels[0]) - 0.5) / npixels_effective[0],
            (_one_to(npixels[1] + 1) - 0.5) / npixels_effective[1],
        ]
        ind_PML_2 = [
            _reverse_one_to(npixels[0]),
            (n + 1) - _reverse_one_to(npixels[1] + 1),
        ]
    else:
        raise ValueError(f'Input argument {direction}BC = "{BC}" is not a supported option.')

    p_PML_1 = [
        _one_to(npixels[0]) / npixels_effective[0],
        _one_to(npixels[1]) / npixels_effective[1],
    ]
    ind_PML_1 = [
        _reverse_one_to(npixels[0]),
        (n + 1) - _reverse_one_to(npixels[1]),
    ]

    ind_PML_E = ind_PML_1
    ind_PML_H = ind_PML_2
    p_PML_E = p_PML_1
    p_PML_H = p_PML_2

    for idx, layer in enumerate(pml):
        if npixels[idx] <= 0:
            continue
        missing = [
            name
            for name in (
                "kappa_max",
                "power_kappa",
                "sigma_max_over_omega",
                "power_sigma",
                "alpha_max_over_omega",
                "power_alpha",
            )
            if getattr(layer, name) is None
        ]
        if missing:
            raise ValueError(f"{direction}PML[{idx}] is missing parameters: {', '.join(missing)}")

        def func_s(p: np.ndarray) -> np.ndarray:
            kappa = 1 + (layer.kappa_max - 1) * p**layer.power_kappa
            sigma_over_omega = layer.sigma_max_over_omega * p**layer.power_sigma
            alpha_over_omega = layer.alpha_max_over_omega * (1 - p) ** layer.power_alpha
            return kappa + sigma_over_omega / (alpha_over_omega - 1j)

        e_idx = ind_PML_E[idx].astype(int) - 1
        h_idx = ind_PML_H[idx].astype(int) - 1
        s_E[e_idx] = func_s(p_PML_E[idx])
        s_H[h_idx] = func_s(p_PML_H[idx])

    ind_PML_E_zero_based = [
        item.astype(int) - 1 if item.size else item.astype(int)
        for item in ind_PML_E
    ]
    return ddx, avg, s_E, s_H, ind_PML_E_zero_based


def build_ave_x_Ex(n_E: int, BC: str | numbers.Number, direction: str) -> sparse.csc_matrix:
    """Build the average matrix that acts on Ex along one direction."""

    if _is_number(BC):
        kLambda = complex(BC)
        BC = "Bloch"
    elif BC == "periodic":
        kLambda = 0j
        BC = "Bloch"
    else:
        kLambda = 0j

    one = np.ones(n_E, dtype=np.complex128)
    if BC == "Bloch":
        return _diag_matrix(
            n_E,
            n_E,
            [
                (1, np.ones(n_E - 1) / 2),
                (0, one / 2),
                (1 - n_E, np.array([np.exp(1j * kLambda) / 2])),
            ],
        )
    if BC == "PEC":
        return _diag_matrix(n_E - 1, n_E, [(1, np.ones(n_E - 1) / 2), (0, np.ones(n_E - 1) / 2)])
    if BC == "PMC":
        return _diag_matrix(n_E + 1, n_E, [(0, one / 2), (-1, one / 2)])
    if BC == "PECPMC":
        return _diag_matrix(n_E, n_E, [(1, np.ones(n_E - 1) / 2), (0, one / 2)])
    if BC == "PMCPEC":
        return _diag_matrix(n_E, n_E, [(0, one / 2), (-1, np.ones(n_E - 1) / 2)])
    raise ValueError(f'Input argument {direction}BC = "{BC}" is not a supported option.')
