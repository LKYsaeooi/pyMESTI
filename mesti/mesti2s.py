"""Two-sided 2D TM scattering wrapper for the Python MESTI port."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import numbers
from typing import Any

import numpy as np

from .channels import mesti_build_channels
from .mesti import mesti
from .types import (
    Channels_one_sided,
    Channels_two_sided,
    Info,
    Opts,
    PML,
    Source_struct,
    Syst,
    channel_index,
    channel_type,
    wavefront,
)


_DN = 0.5


def _as_2d_tm_epsilon(syst: Syst) -> np.ndarray:
    epsilon_xx = np.asarray(syst.epsilon_xx, dtype=np.complex128)
    if epsilon_xx.ndim != 2:
        raise NotImplementedError("Python mesti2s currently supports only 2D TM systems.")
    if epsilon_xx.shape[0] <= 0:
        raise ValueError("syst.epsilon_xx must have at least one y pixel.")
    return epsilon_xx


def _as_real_scalar(value: Any, name: str, allow_none: bool = False) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be a real scalar.")
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"{name} must be a real scalar.")
    scalar = arr.reshape(-1)[0]
    if np.iscomplexobj(arr) and not np.isclose(np.imag(scalar), 0):
        raise ValueError(f"{name} must be real.")
    return float(np.real(scalar))


def _as_nonnegative_int(value: Any, name: str) -> int:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"{name} must be a non-negative integer scalar.")
    scalar = arr.reshape(-1)[0]
    if np.iscomplexobj(arr) and not np.isclose(np.imag(scalar), 0):
        raise ValueError(f"{name} must be a non-negative integer scalar.")
    real_value = float(np.real(scalar))
    if real_value < 0 or not real_value.is_integer():
        raise ValueError(f"{name} must be a non-negative integer scalar.")
    return int(real_value)


def _as_optional_bool(value: Any, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be a boolean scalar, if given.")


def _prepare_opts(output: Any, opts: Opts | None) -> Opts:
    opts = deepcopy(opts) if opts is not None else Opts()
    if opts.prefactor is not None:
        raise ValueError("opts.prefactor is not used in mesti2s(); the -2i prefactor is automatically included.")
    opts.return_field_profile = output is None
    if opts.verbal is None:
        opts.verbal = False
    if opts.clear_memory is None:
        opts.clear_memory = True
    if opts.use_continuous_dispersion is None:
        opts.use_continuous_dispersion = False
    if opts.n0 is None:
        opts.n0 = 0
    if opts.m0 is None:
        opts.m0 = 0
    if opts.return_field_profile:
        if opts.nz_low is None:
            opts.nz_low = 0
        else:
            opts.nz_low = _as_nonnegative_int(opts.nz_low, "opts.nz_low")
        if opts.nz_high is None:
            opts.nz_high = 0
        else:
            opts.nz_high = _as_nonnegative_int(opts.nz_high, "opts.nz_high")
    else:
        opts.nz_low = None
        opts.nz_high = None
    return opts


def _as_pml_list(zpml: PML | list[PML] | None, two_sided: bool) -> tuple[list[PML], list[int]]:
    if zpml is None:
        raise ValueError("syst.zPML must be a PML or list of PML objects for mesti2s.")
    layers = [deepcopy(zpml)] if isinstance(zpml, PML) else [deepcopy(layer) for layer in zpml]
    if not layers or len(layers) > 2:
        raise ValueError("syst.zPML must contain one or two PML objects.")
    if two_sided and len(layers) == 1:
        layers = [deepcopy(layers[0]), deepcopy(layers[0])]
    if (not two_sided) and len(layers) == 2:
        raise ValueError("one-sided 2D TM mesti2s accepts only one zPML object.")

    nz_extra: list[int] = []
    for idx, layer in enumerate(layers):
        if layer.npixels is None:
            raise ValueError(f"syst.zPML[{idx}].npixels must be set.")
        if layer.npixels < 0:
            raise ValueError(f"syst.zPML[{idx}].npixels must be non-negative.")
        spacer = 0 if layer.npixels_spacer is None else int(layer.npixels_spacer)
        if spacer < 0:
            raise ValueError(f"syst.zPML[{idx}].npixels_spacer must be non-negative.")
        layer.direction = "z"
        layer.side = "-" if idx == 0 else "+"
        # Julia's mesti2s pads each side by one source/projection pixel, the
        # absorbing PML pixels, and any user spacer pixels, then clears the
        # spacer before delegating to the lower-level mesti() solver.
        layer.npixels_spacer = None
        nz_extra.append(1 + int(layer.npixels) + spacer)

    if not two_sided:
        layers.append(PML(0, direction="z", side="+"))
        nz_extra.append(0)
    return layers, nz_extra


def _side_low(channels: Channels_one_sided | Channels_two_sided):
    return channels.low if isinstance(channels, Channels_two_sided) else channels


def _side_high(channels: Channels_two_sided):
    if channels.high is None:
        raise ValueError("two-sided channel metadata is missing the high side.")
    return channels.high


def _as_index_array(value: Any, n_prop: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=int).reshape(-1)
    if arr.size and (np.any(arr < 0) or np.any(arr >= n_prop)):
        raise ValueError(f"{name} must contain zero-based indices in [0, {n_prop}).")
    return arr


def _as_wave_matrix(value: Any, n_prop: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2 or arr.shape[0] != n_prop:
        raise ValueError(f"{name} must have shape ({n_prop}, n_wavefronts).")
    return arr


def _parse_input(
    input_spec: channel_type | channel_index | wavefront,
    n_low: int,
    n_high: int | None,
) -> dict[str, Any]:
    two_sided = n_high is not None
    if isinstance(input_spec, channel_type):
        if input_spec.side not in {"low", "high", "both"}:
            raise ValueError('input.side must be "low", "high", or "both".')
        ind_low = np.arange(n_low, dtype=int) if input_spec.side in {"low", "both"} else np.array([], dtype=int)
        if input_spec.side in {"high", "both"}:
            if not two_sided:
                raise ValueError('input.side = "high" requires a two-sided geometry.')
            ind_high = np.arange(n_high, dtype=int)
        else:
            ind_high = np.array([], dtype=int)
        return {"use_indices": True, "low": ind_low, "high": ind_high}

    if isinstance(input_spec, channel_index):
        ind_low = (
            _as_index_array(input_spec.ind_low, n_low, "input.ind_low")
            if input_spec.ind_low is not None
            else np.array([], dtype=int)
        )
        if input_spec.ind_high is not None:
            if not two_sided:
                raise ValueError("input.ind_high requires a two-sided geometry.")
            ind_high = _as_index_array(input_spec.ind_high, n_high, "input.ind_high")
        else:
            ind_high = np.array([], dtype=int)
        if ind_low.size == 0 and ind_high.size == 0:
            raise ValueError("channel_index input must specify ind_low or ind_high.")
        return {"use_indices": True, "low": ind_low, "high": ind_high}

    if not isinstance(input_spec, wavefront):
        raise TypeError("input must be channel_type, channel_index, or wavefront.")

    v_low = (
        _as_wave_matrix(input_spec.v_low, n_low, "input.v_low")
        if input_spec.v_low is not None
        else np.zeros((n_low, 0), dtype=np.complex128)
    )
    if input_spec.v_high is not None:
        if not two_sided:
            raise ValueError("input.v_high requires a two-sided geometry.")
        v_high = _as_wave_matrix(input_spec.v_high, n_high, "input.v_high")
    else:
        v_high = np.zeros((n_high or 0, 0), dtype=np.complex128)
    if v_low.shape[1] == 0 and v_high.shape[1] == 0:
        raise ValueError("wavefront input must specify v_low or v_high.")
    return {"use_indices": False, "low": v_low, "high": v_high}


def _parse_output(
    output_spec: channel_type | channel_index | wavefront,
    n_low: int,
    n_high: int | None,
) -> dict[str, Any]:
    parsed = _parse_input(output_spec, n_low, n_high)
    return parsed


def _selection_count(parsed: dict[str, Any]) -> int:
    if parsed["use_indices"]:
        return len(parsed["low"]) + len(parsed["high"])
    return parsed["low"].shape[1] + parsed["high"].shape[1]


def _ensure_nonempty_selection(parsed: dict[str, Any], name: str) -> None:
    if _selection_count(parsed) == 0:
        raise ValueError(f"{name} selects no propagating channels.")


def _phase_prefactor(side: Any, indices: np.ndarray) -> np.ndarray:
    # The source/projection planes are half a pixel from z=0 or z=L in Julia
    # (`dn = 0.5`), so channel-index paths get this phase/flux factor after
    # the direct solve rather than inside the raw Source_struct block.
    return side.sqrt_nu_prop[indices] * np.exp((-1j * _DN) * side.kzdx_prop[indices])


def _wave_prefactor(side: Any) -> np.ndarray:
    return side.sqrt_nu_prop * np.exp((-1j * _DN) * side.kzdx_prop)


def _build_input_block(f_prop: np.ndarray, side: Any, parsed: dict[str, Any], key: str) -> np.ndarray:
    values = parsed[key]
    if parsed["use_indices"]:
        return f_prop[:, values]
    return f_prop @ (_wave_prefactor(side)[:, np.newaxis] * values)


def _build_output_block(f_prop: np.ndarray, side: Any, parsed: dict[str, Any], key: str) -> np.ndarray:
    values = parsed[key]
    if parsed["use_indices"]:
        return np.conjugate(f_prop[:, values])
    return np.conjugate(
        f_prop @ (np.conjugate(_wave_prefactor(side))[:, np.newaxis] * values)
    )


def _symmetrized_side_expansion(
    side: Any,
    input_indices: np.ndarray,
    output_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conj = np.asarray(side.ind_prop_conj, dtype=int).reshape(-1)
    output_conj = conj[np.asarray(output_indices, dtype=int)]
    expanded = np.unique(np.concatenate([np.asarray(input_indices, dtype=int), output_conj]))
    lookup = {int(channel): pos for pos, channel in enumerate(expanded)}
    input_positions = np.array([lookup[int(channel)] for channel in input_indices], dtype=int)
    output_positions = np.array([lookup[int(channel)] for channel in output_conj], dtype=int)
    return expanded, input_positions, output_positions


def _symmetrized_channel_expansion(
    input_parsed: dict[str, Any],
    output_parsed: dict[str, Any],
    low: Any,
    high: Any | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    low_indices, low_in_pos, low_out_pos = _symmetrized_side_expansion(
        low,
        input_parsed["low"],
        output_parsed["low"],
    )
    if high is None:
        return (
            {"use_indices": True, "low": low_indices, "high": np.array([], dtype=int)},
            low_out_pos,
            low_in_pos,
        )

    high_indices, high_in_pos, high_out_pos = _symmetrized_side_expansion(
        high,
        input_parsed["high"],
        output_parsed["high"],
    )
    return (
        {"use_indices": True, "low": low_indices, "high": high_indices},
        np.concatenate([low_out_pos, len(low_indices) + high_out_pos]),
        np.concatenate([low_in_pos, len(low_indices) + high_in_pos]),
    )


def _can_symmetrize_2d_transverse_operator(yBC: Any) -> bool:
    return not (isinstance(yBC, numbers.Number) and yBC != 0)


def _solver_can_default_to_mumpspy_apf(solver: str | None) -> bool:
    requested = solver.lower() if isinstance(solver, str) else None
    if requested == "mumpspy":
        return True
    if requested in {None, "", "auto", "mumps"}:
        return importlib.util.find_spec("mumpspy") is not None
    return False


def _prefer_mumpspy_apf_for_projected_2d_solve(opts: Opts, output: Any) -> None:
    if output is None or opts.method is not None:
        return
    if _solver_can_default_to_mumpspy_apf(opts.solver):
        opts.method = "APF"


def _surface_pos(ny: int, z_index: int) -> np.ndarray:
    # Python Source_struct positions are zero-based inclusive equivalents of
    # Julia's [1, l, ny, 1] 2D TM surface slices.
    return np.array([0, z_index, ny - 1, z_index], dtype=int)


def _padded_system(
    syst: Syst,
    epsilon_xx: np.ndarray,
    pml_layers: list[PML],
    nz_extra: list[int],
    two_sided: bool,
    yBC: Any,
) -> Syst:
    low = np.full((epsilon_xx.shape[0], nz_extra[0]), syst.epsilon_low, dtype=np.complex128)
    parts = [low, epsilon_xx]
    if two_sided:
        high = np.full((epsilon_xx.shape[0], nz_extra[1]), syst.epsilon_high, dtype=np.complex128)
        parts.append(high)
    padded = deepcopy(syst)
    padded.epsilon_xx = np.concatenate(parts, axis=1)
    padded.epsilon_low = None
    padded.epsilon_high = None
    padded.yBC = yBC
    padded.ky_B = None
    padded.zPML = None
    padded.PML = pml_layers
    padded.zBC = "PEC"
    return padded


def _direct_term_for_side(
    side: Any,
    input_parsed: dict[str, Any],
    output_parsed: dict[str, Any],
    key: str,
) -> np.ndarray:
    phase = np.exp((-1j * 2 * _DN) * side.kzdx_prop)
    if input_parsed["use_indices"]:
        d_side = np.diag(phase)[:, input_parsed[key]]
    else:
        d_side = phase[:, np.newaxis] * input_parsed[key]

    if output_parsed["use_indices"]:
        return d_side[output_parsed[key], :]
    return output_parsed[key].conjugate().T @ d_side


def _subtract_direct_terms(
    S: np.ndarray,
    low: Any,
    high: Any | None,
    input_parsed: dict[str, Any],
    output_parsed: dict[str, Any],
) -> np.ndarray:
    m_out_low = len(output_parsed["low"]) if output_parsed["use_indices"] else output_parsed["low"].shape[1]
    m_in_low = len(input_parsed["low"]) if input_parsed["use_indices"] else input_parsed["low"].shape[1]
    direct = np.zeros_like(S)
    if m_out_low and m_in_low:
        direct[:m_out_low, :m_in_low] = _direct_term_for_side(low, input_parsed, output_parsed, "low")

    if high is not None:
        m_out_high = len(output_parsed["high"]) if output_parsed["use_indices"] else output_parsed["high"].shape[1]
        m_in_high = len(input_parsed["high"]) if input_parsed["use_indices"] else input_parsed["high"].shape[1]
        if m_out_high and m_in_high:
            direct[m_out_low:, m_in_low:] = _direct_term_for_side(
                high,
                input_parsed,
                output_parsed,
                "high",
            )
    return S - direct


def _input_count(parsed: dict[str, Any], key: str) -> int:
    values = parsed[key]
    return len(values) if parsed["use_indices"] else values.shape[1]


def _incident_prop_coefficients(side: Any, parsed: dict[str, Any], key: str, column: int) -> np.ndarray:
    prefactor = np.exp((-1j * _DN) * side.kzdx_prop) / side.sqrt_nu_prop
    if parsed["use_indices"]:
        coeff = np.zeros(side.N_prop, dtype=np.complex128)
        coeff[parsed[key][column]] = prefactor[parsed[key][column]]
        return coeff
    return prefactor * parsed[key][:, column]


def _extend_low_field_profile(
    Ex: np.ndarray,
    channels: Channels_one_sided | Channels_two_sided,
    low: Any,
    high: Any | None,
    input_parsed: dict[str, Any],
    nz_low_extra: int,
) -> np.ndarray:
    u = channels.f_x_m(channels.kydx_all)
    f_prime = np.conjugate(u).T
    l_values = np.arange(-nz_low_extra, 0, dtype=float)
    exp_mikz = np.exp(-1j * low.kzdx_all[:, np.newaxis] * l_values[np.newaxis, :])
    exp_pikz_prop = np.exp(1j * low.kzdx_prop[:, np.newaxis] * l_values[np.newaxis, :])

    m_low = _input_count(input_parsed, "low")
    m_high = _input_count(input_parsed, "high") if high is not None else 0
    Ex_low = np.zeros((Ex.shape[0], nz_low_extra, Ex.shape[2]), dtype=np.complex128)

    for column in range(m_low):
        c = f_prime @ Ex[:, 0, column]
        c_in_prop = _incident_prop_coefficients(low, input_parsed, "low", column)
        c_in = np.zeros_like(c)
        c_in[low.ind_prop] = c_in_prop
        c_out = c - c_in
        Ex_low[:, :, column] = (
            u[:, low.ind_prop] @ (c_in_prop[:, np.newaxis] * exp_pikz_prop)
            + u @ (c_out[:, np.newaxis] * exp_mikz)
        )

    for column in range(m_high):
        out_column = m_low + column
        c_out = f_prime @ Ex[:, 0, out_column]
        Ex_low[:, :, out_column] = u @ (c_out[:, np.newaxis] * exp_mikz)

    return np.concatenate([Ex_low, Ex], axis=1)


def _extend_high_field_profile(
    Ex: np.ndarray,
    channels: Channels_two_sided,
    high: Any,
    input_parsed: dict[str, Any],
    nz_high_extra: int,
) -> np.ndarray:
    u = channels.f_x_m(channels.kydx_all)
    f_prime = np.conjugate(u).T
    l_values = np.arange(1, nz_high_extra + 1, dtype=float)
    exp_pikz = np.exp(1j * high.kzdx_all[:, np.newaxis] * l_values[np.newaxis, :])
    exp_mikz_prop = np.exp(-1j * high.kzdx_prop[:, np.newaxis] * l_values[np.newaxis, :])

    m_low = _input_count(input_parsed, "low")
    m_high = _input_count(input_parsed, "high")
    Ex_high = np.zeros((Ex.shape[0], nz_high_extra, Ex.shape[2]), dtype=np.complex128)
    high_surface = Ex.shape[1] - 1

    for column in range(m_low):
        c_out = f_prime @ Ex[:, high_surface, column]
        Ex_high[:, :, column] = u @ (c_out[:, np.newaxis] * exp_pikz)

    for column in range(m_high):
        out_column = m_low + column
        c = f_prime @ Ex[:, high_surface, out_column]
        c_in_prop = _incident_prop_coefficients(high, input_parsed, "high", column)
        c_in = np.zeros_like(c)
        c_in[high.ind_prop] = c_in_prop
        c_out = c - c_in
        Ex_high[:, :, out_column] = (
            u[:, high.ind_prop] @ (c_in_prop[:, np.newaxis] * exp_mikz_prop)
            + u @ (c_out[:, np.newaxis] * exp_pikz)
        )

    return np.concatenate([Ex, Ex_high], axis=1)


def _crop_field_profile(
    Ex: np.ndarray,
    original_nz: int,
    nz_extra: list[int],
    two_sided: bool,
    opts: Opts,
    channels: Channels_one_sided | Channels_two_sided,
    low: Any,
    high: Any | None,
    input_parsed: dict[str, Any],
) -> np.ndarray:
    nz_remove_low = nz_extra[0] - 1
    nz_remove_high = nz_extra[1] - 1 if two_sided else 0
    Ex = Ex[:, nz_remove_low : Ex.shape[1] - nz_remove_high, :]

    nz_low = int(opts.nz_low or 0)
    nz_high = int(opts.nz_high or 0)

    nz_low_extra = nz_low - 1
    if nz_low_extra == -1:
        Ex = Ex[:, 1:, :]
    elif nz_low_extra > 0:
        Ex = _extend_low_field_profile(Ex, channels, low, high, input_parsed, nz_low_extra)

    if two_sided:
        if high is None or not isinstance(channels, Channels_two_sided):
            raise ValueError("two-sided field cropping requires high-side channels.")
        nz_high_extra = nz_high - 1
        if nz_high_extra == -1:
            Ex = Ex[:, :-1, :]
        elif nz_high_extra > 0:
            Ex = _extend_high_field_profile(Ex, channels, high, input_parsed, nz_high_extra)
    elif nz_high > 0:
        Ex = np.concatenate(
            [Ex, np.zeros((Ex.shape[0], nz_high, Ex.shape[2]), dtype=np.complex128)],
            axis=1,
        )

    expected_nz = nz_low + original_nz + nz_high
    if Ex.shape[1] != expected_nz:
        raise RuntimeError(f"Internal field-profile crop error: got nz={Ex.shape[1]}, expected {expected_nz}.")
    return Ex


def _off_diagonal_components(syst: Syst) -> tuple[Any, ...]:
    return (syst.epsilon_xy, syst.epsilon_xz, syst.epsilon_yx, syst.epsilon_yz, syst.epsilon_zx, syst.epsilon_zy)


def _as_3d_epsilon(syst: Syst) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epsilon_xx = np.asarray(syst.epsilon_xx, dtype=np.complex128)
    if epsilon_xx.ndim != 3:
        raise NotImplementedError("Python 3D mesti2s requires a 3D syst.epsilon_xx array.")
    if syst.epsilon_yy is None or syst.epsilon_zz is None:
        raise ValueError("3D mesti2s requires syst.epsilon_yy and syst.epsilon_zz.")
    epsilon_yy = np.asarray(syst.epsilon_yy, dtype=np.complex128)
    epsilon_zz = np.asarray(syst.epsilon_zz, dtype=np.complex128)
    if epsilon_yy.ndim != 3 or epsilon_zz.ndim != 3:
        raise ValueError("3D mesti2s requires 3D epsilon_yy and epsilon_zz arrays.")
    if min(epsilon_xx.shape + epsilon_yy.shape + epsilon_zz.shape) <= 0:
        raise ValueError("3D epsilon arrays must have positive dimensions.")
    return epsilon_xx, epsilon_yy, epsilon_zz


def _optional_3d_tensor_component(
    component: Any,
    expected_shape: tuple[int, int, int],
    name: str,
) -> np.ndarray | None:
    if component is None:
        return None
    array = np.asarray(component, dtype=np.complex128)
    if array.shape != expected_shape:
        raise ValueError(f"3D mesti2s {name} must have shape {expected_shape}; got {array.shape}.")
    return array


def _as_3d_off_diagonal_epsilon(
    syst: Syst,
    epsilon_xx: np.ndarray,
    epsilon_yy: np.ndarray,
    epsilon_zz: np.ndarray,
) -> dict[str, np.ndarray | None]:
    nx_Ex, ny_Ex, nz_Ex = epsilon_xx.shape
    nx_Ey, ny_Ey, nz_Ey = epsilon_yy.shape
    nx_Ez, ny_Ez, _ = epsilon_zz.shape
    return {
        "epsilon_xy": _optional_3d_tensor_component(syst.epsilon_xy, (nx_Ez, ny_Ez, nz_Ex), "epsilon_xy"),
        "epsilon_xz": _optional_3d_tensor_component(syst.epsilon_xz, (nx_Ey, ny_Ex, nz_Ey), "epsilon_xz"),
        "epsilon_yx": _optional_3d_tensor_component(syst.epsilon_yx, (nx_Ez, ny_Ez, nz_Ey), "epsilon_yx"),
        "epsilon_yz": _optional_3d_tensor_component(syst.epsilon_yz, (nx_Ey, ny_Ex, nz_Ex), "epsilon_yz"),
        "epsilon_zx": _optional_3d_tensor_component(syst.epsilon_zx, (nx_Ey, ny_Ez, nz_Ey), "epsilon_zx"),
        "epsilon_zy": _optional_3d_tensor_component(syst.epsilon_zy, (nx_Ez, ny_Ex, nz_Ex), "epsilon_zy"),
    }


def _as_3d_boundary_phases(
    syst: Syst,
    epsilon_xx: np.ndarray,
    epsilon_yy: np.ndarray,
) -> tuple[Any, Any]:
    nx_Ex = epsilon_xx.shape[0]
    ny_Ey = epsilon_yy.shape[1]
    if syst.kx_B is not None:
        xBC = syst.kx_B * (nx_Ex * syst.dx)
    elif syst.xBC is not None:
        xBC = syst.xBC
    else:
        raise ValueError('Input argument syst must have non-empty field "xBC" for 3D mesti2s.')
    if syst.ky_B is not None:
        yBC = syst.ky_B * (ny_Ey * syst.dx)
    elif syst.yBC is not None:
        yBC = syst.yBC
    else:
        raise ValueError('Input argument syst must have non-empty field "yBC".')
    return xBC, yBC


def _as_index_array_3d(value: Any, n_prop: int, name: str) -> np.ndarray:
    if value is None:
        return np.array([], dtype=int)
    return _as_index_array(value, n_prop, name)


def _as_wave_matrix_3d(value: Any, n_prop: int, name: str) -> np.ndarray:
    if value is None:
        return np.zeros((n_prop, 0), dtype=np.complex128)
    return _as_wave_matrix(value, n_prop, name)


def _parse_3d_selector(
    spec: channel_type | channel_index | wavefront,
    n_low: int,
    n_high: int | None,
    name: str,
) -> dict[str, Any]:
    two_sided = n_high is not None
    if isinstance(spec, channel_type):
        if spec.side not in {"low", "high", "both"}:
            raise ValueError(f'{name}.side must be "low", "high", or "both".')
        polarization = "both" if spec.polarization is None else spec.polarization
        if polarization not in {"s", "p", "both"}:
            raise ValueError(f'{name}.polarization must be "s", "p", or "both".')
        empty = np.array([], dtype=int)
        low_s = np.arange(n_low, dtype=int) if spec.side in {"low", "both"} and polarization in {"s", "both"} else empty
        low_p = np.arange(n_low, dtype=int) if spec.side in {"low", "both"} and polarization in {"p", "both"} else empty
        if spec.side in {"high", "both"}:
            if not two_sided:
                raise ValueError(f'{name}.side = "high" requires a two-sided geometry.')
            high_s = np.arange(n_high, dtype=int) if polarization in {"s", "both"} else empty
            high_p = np.arange(n_high, dtype=int) if polarization in {"p", "both"} else empty
        else:
            high_s = empty
            high_p = empty
        return {
            "use_indices": True,
            "low_s": low_s,
            "low_p": low_p,
            "high_s": high_s,
            "high_p": high_p,
        }

    if isinstance(spec, channel_index):
        high_s = _as_index_array_3d(spec.ind_high_s, n_high or 0, f"{name}.ind_high_s")
        high_p = _as_index_array_3d(spec.ind_high_p, n_high or 0, f"{name}.ind_high_p")
        if (high_s.size or high_p.size) and not two_sided:
            raise ValueError(f"{name}.ind_high_s and {name}.ind_high_p require a two-sided geometry.")
        parsed = {
            "use_indices": True,
            "low_s": _as_index_array_3d(spec.ind_low_s, n_low, f"{name}.ind_low_s"),
            "low_p": _as_index_array_3d(spec.ind_low_p, n_low, f"{name}.ind_low_p"),
            "high_s": high_s,
            "high_p": high_p,
        }
        if _selection_count_3d(parsed) == 0:
            raise ValueError(f"{name} channel_index must specify at least one 3D s/p channel field.")
        return parsed

    if not isinstance(spec, wavefront):
        raise TypeError(f"{name} must be channel_type, channel_index, or wavefront.")

    if (spec.v_high_s is not None or spec.v_high_p is not None) and not two_sided:
        raise ValueError(f"{name}.v_high_s and {name}.v_high_p require a two-sided geometry.")
    parsed = {
        "use_indices": False,
        "low_s": _as_wave_matrix_3d(spec.v_low_s, n_low, f"{name}.v_low_s"),
        "low_p": _as_wave_matrix_3d(spec.v_low_p, n_low, f"{name}.v_low_p"),
        "high_s": _as_wave_matrix_3d(spec.v_high_s, n_high or 0, f"{name}.v_high_s"),
        "high_p": _as_wave_matrix_3d(spec.v_high_p, n_high or 0, f"{name}.v_high_p"),
    }
    if _selection_count_3d(parsed) == 0:
        raise ValueError(f"{name} wavefront must specify at least one 3D s/p wavefront field.")
    return parsed


def _selection_count_3d(parsed: dict[str, Any]) -> int:
    keys = ("low_s", "low_p", "high_s", "high_p")
    if parsed["use_indices"]:
        return sum(len(parsed[key]) for key in keys)
    return sum(parsed[key].shape[1] for key in keys)


def _ensure_nonempty_selection_3d(parsed: dict[str, Any], name: str) -> None:
    if _selection_count_3d(parsed) == 0:
        raise ValueError(f"{name} selects no propagating channels.")


def _side_count_3d(parsed: dict[str, Any], side: str) -> int:
    if parsed["use_indices"]:
        return len(parsed[f"{side}_s"]) + len(parsed[f"{side}_p"])
    return parsed[f"{side}_s"].shape[1] + parsed[f"{side}_p"].shape[1]


def _ordered_count_3d(parsed: dict[str, Any]) -> int:
    return _side_count_3d(parsed, "low") + _side_count_3d(parsed, "high")


def _replace_nan(values: np.ndarray, replacement: complex) -> np.ndarray:
    result = np.asarray(values, dtype=np.complex128).copy()
    result[np.isnan(result)] = replacement
    return result


def _polarization_coefficients(side: Any) -> dict[str, np.ndarray]:
    kappa_x = np.sin(side.kxdx_prop / 2)
    kappa_y = np.sin(side.kydx_prop / 2)
    kappa_z = np.sin(side.kzdx_prop / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator_s = np.sqrt(kappa_x**2 + kappa_y**2)
        alpha_x_s = -kappa_y / denominator_s
        alpha_y_s = kappa_x / denominator_s
        denominator_p = np.sqrt(
            np.abs(kappa_x * kappa_z) ** 2
            + np.abs(kappa_y * kappa_z) ** 2
            + np.abs(kappa_x**2 + kappa_y**2) ** 2
        )
        alpha_x_p = kappa_x * kappa_z / denominator_p
        alpha_y_p = kappa_y * kappa_z / denominator_p
        alpha_z_p = -(kappa_x**2 + kappa_y**2) / denominator_p
    return {
        "x_s": _replace_nan(alpha_x_s, 0),
        "y_s": _replace_nan(alpha_y_s, 1),
        "x_p": _replace_nan(alpha_x_p, 1),
        "y_p": _replace_nan(alpha_y_p, 0),
        "z_p": _replace_nan(alpha_z_p, 0),
    }


def _product_modes(n_func: Any, m_func: Any, kxdx: np.ndarray, kydx: np.ndarray) -> np.ndarray:
    f_n = np.asarray(n_func(kxdx), dtype=np.complex128)
    f_m = np.asarray(m_func(kydx), dtype=np.complex128)
    if f_n.shape[1] != f_m.shape[1]:
        raise RuntimeError("Internal 3D channel mode count mismatch.")
    # Each column is a transverse channel profile flattened with x fastest,
    # matching Julia's reshape([x, y], :) convention for surface slices.
    values = f_n[:, np.newaxis, :] * f_m[np.newaxis, :, :]
    return values.reshape((f_n.shape[0] * f_m.shape[0], f_n.shape[1]), order="F")


def _cartesian_product_modes(n_func: Any, m_func: Any, kxdx: np.ndarray, kydx: np.ndarray) -> np.ndarray:
    f_n = np.asarray(n_func(kxdx), dtype=np.complex128)
    f_m = np.asarray(m_func(kydx), dtype=np.complex128)
    values = f_n[:, np.newaxis, :, np.newaxis] * f_m[np.newaxis, :, np.newaxis, :]
    return values.reshape((f_n.shape[0] * f_m.shape[0], f_n.shape[1] * f_m.shape[1]), order="F")


def _surface_modes_3d(channels: Channels_one_sided | Channels_two_sided, side: Any) -> dict[str, np.ndarray]:
    return {
        "Ex": _product_modes(channels.f_x_n, channels.f_x_m, side.kxdx_prop, side.kydx_prop),
        "Ey": _product_modes(channels.f_y_n, channels.f_y_m, side.kxdx_prop, side.kydx_prop),
        "dEz_dx": _product_modes(channels.df_z_n, channels.f_z_m, side.kxdx_prop, side.kydx_prop),
        "dEz_dy": _product_modes(channels.f_z_n, channels.df_z_m, side.kxdx_prop, side.kydx_prop),
    }


def _all_surface_modes_3d(channels: Channels_one_sided | Channels_two_sided) -> dict[str, np.ndarray]:
    return {
        "Ex": _cartesian_product_modes(channels.f_x_n, channels.f_x_m, channels.kxdx_all, channels.kydx_all),
        "Ey": _cartesian_product_modes(channels.f_y_n, channels.f_y_m, channels.kxdx_all, channels.kydx_all),
        "Ez": _cartesian_product_modes(channels.f_z_n, channels.f_z_m, channels.kxdx_all, channels.kydx_all),
    }


def _surface_data(matrix: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return matrix.reshape((nx, ny, 1, matrix.shape[1]), order="F")


def _build_input_surface_3d(
    modes: dict[str, np.ndarray],
    side: Any,
    alpha: dict[str, np.ndarray],
    parsed: dict[str, Any],
    key: str,
    shape_ex: tuple[int, int],
    shape_ey: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if parsed["use_indices"]:
        ind_s = parsed[f"{key}_s"]
        ind_p = parsed[f"{key}_p"]
        sqrt_s = side.sqrt_nu_prop[ind_s][np.newaxis, :]
        sqrt_p = side.sqrt_nu_prop[ind_p][np.newaxis, :]
        dz_weight = (np.cos(side.kzdx_prop[ind_p] / 2) / side.sqrt_nu_prop[ind_p])[np.newaxis, :]

        B_s_Ex = modes["Ex"][:, ind_s] * sqrt_s * alpha["x_s"][ind_s][np.newaxis, :]
        B_s_Ey = modes["Ey"][:, ind_s] * sqrt_s * alpha["y_s"][ind_s][np.newaxis, :]
        B_p_Ex = modes["Ex"][:, ind_p] * sqrt_p * alpha["x_p"][ind_p][np.newaxis, :]
        B_p_Ey = modes["Ey"][:, ind_p] * sqrt_p * alpha["y_p"][ind_p][np.newaxis, :]
        B_p_dx = modes["dEz_dx"][:, ind_p] * dz_weight * alpha["z_p"][ind_p][np.newaxis, :]
        B_p_dy = modes["dEz_dy"][:, ind_p] * dz_weight * alpha["z_p"][ind_p][np.newaxis, :]
    else:
        v_s = parsed[f"{key}_s"]
        v_p = parsed[f"{key}_p"]
        phase = side.sqrt_nu_prop * np.exp((-1j * _DN) * side.kzdx_prop)
        dz_phase = np.cos(side.kzdx_prop / 2) * np.exp((-1j * _DN) * side.kzdx_prop) / side.sqrt_nu_prop

        B_s_Ex = modes["Ex"] @ (phase[:, np.newaxis] * (v_s * alpha["x_s"][:, np.newaxis]))
        B_s_Ey = modes["Ey"] @ (phase[:, np.newaxis] * (v_s * alpha["y_s"][:, np.newaxis]))
        B_p_Ex = modes["Ex"] @ (phase[:, np.newaxis] * (v_p * alpha["x_p"][:, np.newaxis]))
        B_p_Ey = modes["Ey"] @ (phase[:, np.newaxis] * (v_p * alpha["y_p"][:, np.newaxis]))
        B_p_dx = modes["dEz_dx"] @ (dz_phase[:, np.newaxis] * (v_p * alpha["z_p"][:, np.newaxis]))
        B_p_dy = modes["dEz_dy"] @ (dz_phase[:, np.newaxis] * (v_p * alpha["z_p"][:, np.newaxis]))

    B_Ex = np.hstack([B_s_Ex, B_p_Ex + 1j * B_p_dx])
    B_Ey = np.hstack([B_s_Ey, B_p_Ey + 1j * B_p_dy])
    return _surface_data(B_Ex, *shape_ex), _surface_data(B_Ey, *shape_ey)


def _build_output_surface_3d(
    modes: dict[str, np.ndarray],
    side: Any,
    alpha: dict[str, np.ndarray],
    parsed: dict[str, Any],
    key: str,
    shape_ex: tuple[int, int],
    shape_ey: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if parsed["use_indices"]:
        ind_s = parsed[f"{key}_s"]
        ind_p = parsed[f"{key}_p"]
        sqrt_s = side.sqrt_nu_prop[ind_s][np.newaxis, :]
        sqrt_p = side.sqrt_nu_prop[ind_p][np.newaxis, :]
        dz_weight = (np.cos(side.kzdx_prop[ind_p] / 2) / side.sqrt_nu_prop[ind_p])[np.newaxis, :]

        C_s_Ex = np.conjugate(modes["Ex"][:, ind_s]) * sqrt_s * np.conjugate(alpha["x_s"][ind_s])[np.newaxis, :]
        C_s_Ey = np.conjugate(modes["Ey"][:, ind_s]) * sqrt_s * np.conjugate(alpha["y_s"][ind_s])[np.newaxis, :]
        C_p_Ex = np.conjugate(modes["Ex"][:, ind_p]) * sqrt_p * np.conjugate(alpha["x_p"][ind_p])[np.newaxis, :]
        C_p_Ey = np.conjugate(modes["Ey"][:, ind_p]) * sqrt_p * np.conjugate(alpha["y_p"][ind_p])[np.newaxis, :]
        C_p_dx = np.conjugate(modes["dEz_dx"][:, ind_p]) * dz_weight * np.conjugate(alpha["z_p"][ind_p])[np.newaxis, :]
        C_p_dy = np.conjugate(modes["dEz_dy"][:, ind_p]) * dz_weight * np.conjugate(alpha["z_p"][ind_p])[np.newaxis, :]
    else:
        v_s = parsed[f"{key}_s"]
        v_p = parsed[f"{key}_p"]
        phase = np.conjugate(side.sqrt_nu_prop * np.exp((-1j * _DN) * side.kzdx_prop))
        dz_phase = np.conjugate(
            np.cos(side.kzdx_prop / 2) * np.exp((-1j * _DN) * side.kzdx_prop) / side.sqrt_nu_prop
        )

        C_s_Ex = np.conjugate(modes["Ex"] @ (phase[:, np.newaxis] * (v_s * alpha["x_s"][:, np.newaxis])))
        C_s_Ey = np.conjugate(modes["Ey"] @ (phase[:, np.newaxis] * (v_s * alpha["y_s"][:, np.newaxis])))
        C_p_Ex = np.conjugate(modes["Ex"] @ (phase[:, np.newaxis] * (v_p * alpha["x_p"][:, np.newaxis])))
        C_p_Ey = np.conjugate(modes["Ey"] @ (phase[:, np.newaxis] * (v_p * alpha["y_p"][:, np.newaxis])))
        C_p_dx = np.conjugate(modes["dEz_dx"] @ (dz_phase[:, np.newaxis] * (v_p * alpha["z_p"][:, np.newaxis])))
        C_p_dy = np.conjugate(modes["dEz_dy"] @ (dz_phase[:, np.newaxis] * (v_p * alpha["z_p"][:, np.newaxis])))

    C_Ex = np.hstack([C_s_Ex, C_p_Ex - 1j * C_p_dx])
    C_Ey = np.hstack([C_s_Ey, C_p_Ey - 1j * C_p_dy])
    return _surface_data(C_Ex, *shape_ex), _surface_data(C_Ey, *shape_ey)


def _surface_pos_3d(nx: int, ny: int, z_index: int) -> np.ndarray:
    # Python 3D Source_struct positions are zero-based inclusive endpoints;
    # the z index is Julia's l_low/l_high surface translated from one-based.
    return np.array([0, 0, z_index, nx - 1, ny - 1, z_index], dtype=int)


def _empty_component_source() -> Source_struct:
    return Source_struct(isempty=True)


def _padded_system_3d(
    syst: Syst,
    epsilon_xx: np.ndarray,
    epsilon_yy: np.ndarray,
    epsilon_zz: np.ndarray,
    off_diagonal: dict[str, np.ndarray | None],
    pml_layers: list[PML],
    nz_extra: list[int],
    two_sided: bool,
    xBC: Any,
    yBC: Any,
) -> Syst:
    padded = deepcopy(syst)

    def pad_component(epsilon: np.ndarray) -> np.ndarray:
        low = np.full((*epsilon.shape[:2], nz_extra[0]), syst.epsilon_low, dtype=np.complex128)
        parts = [low, epsilon]
        if two_sided:
            high = np.full((*epsilon.shape[:2], nz_extra[1]), syst.epsilon_high, dtype=np.complex128)
            parts.append(high)
        return np.concatenate(parts, axis=2)

    padded.epsilon_xx = pad_component(epsilon_xx)
    padded.epsilon_yy = pad_component(epsilon_yy)
    padded.epsilon_zz = pad_component(epsilon_zz)
    for name, component in off_diagonal.items():
        if component is None:
            setattr(padded, name, None)
            continue
        fill_value = 0.0 if two_sided else syst.epsilon_low
        low = np.full((*component.shape[:2], nz_extra[0]), fill_value, dtype=np.complex128)
        parts = [low, component]
        if two_sided:
            high = np.zeros((*component.shape[:2], nz_extra[1]), dtype=np.complex128)
            parts.append(high)
        # Julia pads two-sided off-diagonal homogeneous slabs with zeros, but
        # its one-sided branch fills the low slab with epsilon_low.
        setattr(padded, name, np.concatenate(parts, axis=2))
    padded.epsilon_low = None
    padded.epsilon_high = None
    padded.xBC = xBC
    padded.yBC = yBC
    padded.kx_B = None
    padded.ky_B = None
    padded.zPML = None
    padded.PML = pml_layers
    padded.zBC = "PEC"
    return padded


def _post_index_prefactor_3d(side: Any, ind_s: np.ndarray, ind_p: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.exp((-1j * _DN) * side.kzdx_prop[ind_s]),
            np.exp((-1j * _DN) * side.kzdx_prop[ind_p]),
        ]
    )


def _input_prefactor_3d(parsed: dict[str, Any], low: Any, high: Any | None) -> np.ndarray:
    prefactor = _post_index_prefactor_3d(low, parsed["low_s"], parsed["low_p"])
    if high is not None:
        prefactor = np.concatenate(
            [prefactor, _post_index_prefactor_3d(high, parsed["high_s"], parsed["high_p"])]
        )
    return prefactor


def _direct_side_3d(side: Any, input_parsed: dict[str, Any], output_parsed: dict[str, Any], key: str) -> np.ndarray:
    phase = np.exp((-1j * 2 * _DN) * side.kzdx_prop)
    phase2 = np.concatenate([phase, phase])
    n_prop = side.N_prop
    if input_parsed["use_indices"]:
        cols = np.concatenate([input_parsed[f"{key}_s"], n_prop + input_parsed[f"{key}_p"]])
        d_side = np.diag(phase2)[:, cols]
    else:
        v_s = input_parsed[f"{key}_s"]
        v_p = input_parsed[f"{key}_p"]
        d_side = phase2[:, np.newaxis] * np.block(
            [
                [v_s, np.zeros((n_prop, v_p.shape[1]), dtype=np.complex128)],
                [np.zeros((n_prop, v_s.shape[1]), dtype=np.complex128), v_p],
            ]
        )

    if output_parsed["use_indices"]:
        rows = np.concatenate([output_parsed[f"{key}_s"], n_prop + output_parsed[f"{key}_p"]])
        return d_side[rows, :]
    v_s = output_parsed[f"{key}_s"]
    v_p = output_parsed[f"{key}_p"]
    v_out = np.block(
        [
            [v_s, np.zeros((n_prop, v_p.shape[1]), dtype=np.complex128)],
            [np.zeros((n_prop, v_s.shape[1]), dtype=np.complex128), v_p],
        ]
    )
    return v_out.conjugate().T @ d_side


def _subtract_direct_terms_3d(
    S: np.ndarray,
    low: Any,
    high: Any | None,
    input_parsed: dict[str, Any],
    output_parsed: dict[str, Any],
) -> np.ndarray:
    m_out_low = _side_count_3d(output_parsed, "low")
    m_in_low = _side_count_3d(input_parsed, "low")
    direct = np.zeros_like(S)
    if m_out_low and m_in_low:
        direct[:m_out_low, :m_in_low] = _direct_side_3d(low, input_parsed, output_parsed, "low")

    if high is not None:
        m_out_high = _side_count_3d(output_parsed, "high")
        m_in_high = _side_count_3d(input_parsed, "high")
        if m_out_high and m_in_high:
            direct[m_out_low:, m_in_low:] = _direct_side_3d(high, input_parsed, output_parsed, "high")
    return S - direct


def _all_polarization_coefficients_3d(
    channels: Channels_one_sided | Channels_two_sided,
    side: Any,
    direction: str,
) -> dict[str, np.ndarray]:
    kxdx_all = np.asarray(channels.kxdx_all, dtype=np.complex128).reshape(-1)
    kydx_all = np.asarray(channels.kydx_all, dtype=np.complex128).reshape(-1)
    kappa_x = np.sin(np.tile(kxdx_all, kydx_all.size) / 2)
    kappa_y = np.sin(np.repeat(kydx_all, kxdx_all.size) / 2)
    kappa_z = np.sin(np.asarray(side.kzdx_all, dtype=np.complex128).reshape(-1) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator_s = np.sqrt(kappa_x**2 + kappa_y**2)
        alpha_x_s = -kappa_y / denominator_s
        alpha_y_s = kappa_x / denominator_s
        denominator_p = np.sqrt(
            np.abs(kappa_x * kappa_z) ** 2
            + np.abs(kappa_y * kappa_z) ** 2
            + np.abs(kappa_x**2 + kappa_y**2) ** 2
        )
        alpha_x_p = kappa_x * kappa_z / denominator_p
        alpha_y_p = kappa_y * kappa_z / denominator_p
        alpha_z_sign = 1 if direction == "low" else -1
        alpha_z_p = alpha_z_sign * (kappa_x**2 + kappa_y**2) / denominator_p
    return {
        "x_s": _replace_nan(alpha_x_s, 0),
        "y_s": _replace_nan(alpha_y_s, 1),
        "z_s": np.zeros_like(kappa_z, dtype=np.complex128),
        "x_p": _replace_nan(alpha_x_p, 1),
        "y_p": _replace_nan(alpha_y_p, 0),
        "z_p": _replace_nan(alpha_z_p, 0),
    }


def _basis_matrix_3d(modes: dict[str, np.ndarray], alpha: dict[str, np.ndarray], polarization: str) -> np.ndarray:
    suffix = f"_{polarization}"
    return np.vstack(
        [
            modes["Ex"] * alpha[f"x{suffix}"][np.newaxis, :],
            modes["Ey"] * alpha[f"y{suffix}"][np.newaxis, :],
            modes["Ez"] * alpha[f"z{suffix}"][np.newaxis, :],
        ]
    )


def _field_stack_3d(Ex_slice: np.ndarray, Ey_slice: np.ndarray, Ez_slice: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(Ex_slice, dtype=np.complex128).reshape(-1, order="F"),
            np.asarray(Ey_slice, dtype=np.complex128).reshape(-1, order="F"),
            np.asarray(Ez_slice, dtype=np.complex128).reshape(-1, order="F"),
        ]
    )


def _project_surface_3d(
    Ex_slice: np.ndarray,
    Ey_slice: np.ndarray,
    Ez_slice: np.ndarray,
    basis_s: np.ndarray,
    basis_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stack = _field_stack_3d(Ex_slice, Ey_slice, Ez_slice)
    return basis_s.conjugate().T @ stack, basis_p.conjugate().T @ stack


def _synthesize_layers_3d(
    modes: dict[str, np.ndarray],
    alpha: dict[str, np.ndarray],
    coeff_s: np.ndarray,
    coeff_p: np.ndarray,
    shape_ex: tuple[int, int],
    shape_ey: tuple[int, int],
    shape_ez: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ex = (modes["Ex"] * alpha["x_s"][np.newaxis, :]) @ coeff_s
    Ex += (modes["Ex"] * alpha["x_p"][np.newaxis, :]) @ coeff_p
    Ey = (modes["Ey"] * alpha["y_s"][np.newaxis, :]) @ coeff_s
    Ey += (modes["Ey"] * alpha["y_p"][np.newaxis, :]) @ coeff_p
    Ez = (modes["Ez"] * alpha["z_s"][np.newaxis, :]) @ coeff_s
    Ez += (modes["Ez"] * alpha["z_p"][np.newaxis, :]) @ coeff_p
    nz = coeff_s.shape[1]
    return (
        Ex.reshape((*shape_ex, nz), order="F"),
        Ey.reshape((*shape_ey, nz), order="F"),
        Ez.reshape((*shape_ez, nz), order="F"),
    )


def _input_column_iter_3d(parsed: dict[str, Any]):
    column = 0
    for side in ("low", "high"):
        for polarization in ("s", "p"):
            value = parsed[f"{side}_{polarization}"]
            count = len(value) if parsed["use_indices"] else value.shape[1]
            for local_index in range(count):
                yield column, side, polarization, local_index
                column += 1


def _incident_coefficients_3d(
    parsed: dict[str, Any],
    side: Any,
    side_key: str,
    polarization: str,
    local_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_all = side.kzdx_all.size
    c_s = np.zeros(n_all, dtype=np.complex128)
    c_p = np.zeros(n_all, dtype=np.complex128)
    if parsed["use_indices"]:
        prop_index = int(parsed[f"{side_key}_{polarization}"][local_index])
        all_index = int(side.ind_prop[prop_index])
        value = np.exp((-1j * _DN) * side.kzdx_prop[prop_index]) / side.sqrt_nu_prop[prop_index]
        if polarization == "s":
            c_s[all_index] = value
        else:
            c_p[all_index] = value
        return c_s, c_p

    values = (
        np.exp((-1j * _DN) * side.kzdx_prop)
        / side.sqrt_nu_prop
        * parsed[f"{side_key}_{polarization}"][:, local_index]
    )
    if polarization == "s":
        c_s[side.ind_prop] = values
    else:
        c_p[side.ind_prop] = values
    return c_s, c_p


def _extend_low_field_profile_3d(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
    channels: Channels_one_sided | Channels_two_sided,
    input_parsed: dict[str, Any],
    modes: dict[str, np.ndarray],
    alpha: dict[str, np.ndarray],
    nz_low_extra: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = _side_low(channels)
    basis_s = _basis_matrix_3d(modes, alpha, "s")
    basis_p = _basis_matrix_3d(modes, alpha, "p")
    l_values = np.arange(-nz_low_extra, 0, dtype=float)
    phase = low.kzdx_all[:, np.newaxis] * l_values[np.newaxis, :]
    exp_pikz = np.exp(1j * phase)
    exp_mikz = np.exp(-1j * phase)
    shape_ex = Ex.shape[:2]
    shape_ey = Ey.shape[:2]
    shape_ez = Ez.shape[:2]
    Ex_low = np.zeros((*shape_ex, nz_low_extra, Ex.shape[3]), dtype=np.complex128)
    Ey_low = np.zeros((*shape_ey, nz_low_extra, Ey.shape[3]), dtype=np.complex128)
    Ez_low = np.zeros((*shape_ez, nz_low_extra, Ez.shape[3]), dtype=np.complex128)

    for column, side_key, polarization, local_index in _input_column_iter_3d(input_parsed):
        c_s, c_p = _project_surface_3d(Ex[:, :, 0, column], Ey[:, :, 0, column], Ez[:, :, 0, column], basis_s, basis_p)
        if side_key == "low":
            c_in_s, c_in_p = _incident_coefficients_3d(input_parsed, low, "low", polarization, local_index)
        else:
            c_in_s = np.zeros_like(c_s)
            c_in_p = np.zeros_like(c_p)
        coeff_s = c_in_s[:, np.newaxis] * exp_pikz + (c_s - c_in_s)[:, np.newaxis] * exp_mikz
        coeff_p = c_in_p[:, np.newaxis] * exp_pikz + (c_p - c_in_p)[:, np.newaxis] * exp_mikz
        Ex_low[:, :, :, column], Ey_low[:, :, :, column], Ez_low[:, :, :, column] = _synthesize_layers_3d(
            modes, alpha, coeff_s, coeff_p, shape_ex, shape_ey, shape_ez
        )

    return (
        np.concatenate([Ex_low, Ex], axis=2),
        np.concatenate([Ey_low, Ey], axis=2),
        np.concatenate([Ez_low, Ez], axis=2),
    )


def _extend_high_field_profile_3d(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
    channels: Channels_two_sided,
    input_parsed: dict[str, Any],
    modes: dict[str, np.ndarray],
    alpha: dict[str, np.ndarray],
    nz_high_extra: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high = channels.high
    basis_s = _basis_matrix_3d(modes, alpha, "s")
    basis_p = _basis_matrix_3d(modes, alpha, "p")
    l_values = np.arange(1, nz_high_extra + 1, dtype=float)
    phase = high.kzdx_all[:, np.newaxis] * l_values[np.newaxis, :]
    exp_pikz = np.exp(1j * phase)
    exp_mikz = np.exp(-1j * phase)
    ex_high = Ex.shape[2] - 1
    ey_high = Ey.shape[2] - 1
    ez_high = ex_high + 1
    if ez_high >= Ez.shape[2]:
        raise RuntimeError("Internal 3D field extension requires an Ez high-side surface slice.")
    shape_ex = Ex.shape[:2]
    shape_ey = Ey.shape[:2]
    shape_ez = Ez.shape[:2]
    Ex_high = np.zeros((*shape_ex, nz_high_extra, Ex.shape[3]), dtype=np.complex128)
    Ey_high = np.zeros((*shape_ey, nz_high_extra, Ey.shape[3]), dtype=np.complex128)
    Ez_high = np.zeros((*shape_ez, nz_high_extra, Ez.shape[3]), dtype=np.complex128)

    for column, side_key, polarization, local_index in _input_column_iter_3d(input_parsed):
        c_s, c_p = _project_surface_3d(
            Ex[:, :, ex_high, column],
            Ey[:, :, ey_high, column],
            Ez[:, :, ez_high, column],
            basis_s,
            basis_p,
        )
        if side_key == "high":
            c_in_s, c_in_p = _incident_coefficients_3d(input_parsed, high, "high", polarization, local_index)
        else:
            c_in_s = np.zeros_like(c_s)
            c_in_p = np.zeros_like(c_p)
        coeff_s = c_in_s[:, np.newaxis] * exp_mikz + (c_s - c_in_s)[:, np.newaxis] * exp_pikz
        coeff_p = c_in_p[:, np.newaxis] * exp_mikz + (c_p - c_in_p)[:, np.newaxis] * exp_pikz
        Ex_high[:, :, :, column], Ey_high[:, :, :, column], Ez_high[:, :, :, column] = _synthesize_layers_3d(
            modes, alpha, coeff_s, coeff_p, shape_ex, shape_ey, shape_ez
        )

    return (
        np.concatenate([Ex, Ex_high], axis=2),
        np.concatenate([Ey, Ey_high], axis=2),
        np.concatenate([Ez, Ez_high], axis=2),
    )


def _crop_field_profile_3d(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
    shapes: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    nz_extra: list[int],
    opts: Opts,
    channels: Channels_one_sided | Channels_two_sided,
    input_parsed: dict[str, Any],
    two_sided: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nz_low = int(opts.nz_low or 0)
    nz_high = int(opts.nz_high or 0)

    low_start = nz_extra[0] - 1
    high_trim = nz_extra[1] - 1 if two_sided else 0
    high_stop_ex = Ex.shape[2] - high_trim if high_trim else Ex.shape[2]
    high_stop_ey = Ey.shape[2] - high_trim if high_trim else Ey.shape[2]
    high_stop_ez = Ez.shape[2] - high_trim if high_trim else Ez.shape[2]
    Ex = Ex[:, :, low_start:high_stop_ex, :]
    Ey = Ey[:, :, low_start:high_stop_ey, :]
    Ez = Ez[:, :, low_start:high_stop_ez, :]

    modes = None
    alpha_low = None

    nz_low_extra = nz_low - 1
    if nz_low_extra == -1:
        Ex = Ex[:, :, 1:, :]
        Ey = Ey[:, :, 1:, :]
        Ez = Ez[:, :, 1:, :]
    elif nz_low_extra > 0:
        modes = _all_surface_modes_3d(channels)
        alpha_low = _all_polarization_coefficients_3d(channels, _side_low(channels), "low")
        # The added homogeneous low-side pixels are reconstructed from the
        # retained source surface using the complete transverse mode basis.
        Ex, Ey, Ez = _extend_low_field_profile_3d(
            Ex, Ey, Ez, channels, input_parsed, modes, alpha_low, nz_low_extra
        )

    if two_sided:
        if not isinstance(channels, Channels_two_sided):
            raise RuntimeError("Internal 3D field extension expected two-sided channels.")
        nz_high_extra = nz_high - 1
        if nz_high_extra == -1:
            Ex = Ex[:, :, :-1, :]
            Ey = Ey[:, :, :-1, :]
            Ez = Ez[:, :, :-1, :]
        elif nz_high_extra > 0:
            if modes is None:
                modes = _all_surface_modes_3d(channels)
            alpha_high = _all_polarization_coefficients_3d(channels, channels.high, "high")
            Ex, Ey, Ez = _extend_high_field_profile_3d(
                Ex, Ey, Ez, channels, input_parsed, modes, alpha_high, nz_high_extra
            )
    elif nz_high > 0:
        Ex = np.concatenate([Ex, np.zeros((*Ex.shape[:2], nz_high, Ex.shape[3]), dtype=np.complex128)], axis=2)
        Ey = np.concatenate([Ey, np.zeros((*Ey.shape[:2], nz_high, Ey.shape[3]), dtype=np.complex128)], axis=2)
        Ez = np.concatenate([Ez, np.zeros((*Ez.shape[:2], nz_high, Ez.shape[3]), dtype=np.complex128)], axis=2)

    expected = (
        nz_low + shapes[0][2] + nz_high,
        nz_low + shapes[1][2] + nz_high,
        nz_low + shapes[2][2] + nz_high,
    )
    if (Ex.shape[2], Ey.shape[2], Ez.shape[2]) != expected:
        raise RuntimeError(
            "Internal 3D field-profile crop error: "
            f"got nz={(Ex.shape[2], Ey.shape[2], Ez.shape[2])}, expected {expected}."
        )
    return Ex, Ey, Ez


def _validate_3d_field_extension_shapes(
    shapes: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    two_sided: bool,
) -> None:
    if two_sided and shapes[2][2] < shapes[0][2] + 1:
        raise NotImplementedError(
            "3D mesti2s nz_high extension requires epsilon_zz to include the high-side Yee surface."
        )


def _mesti2s_3d_diagonal(
    syst: Syst,
    input: channel_type | channel_index | wavefront,
    output: channel_type | channel_index | wavefront | Opts | None = None,
    opts: Opts | None = None,
) -> tuple[Any, ...]:
    if isinstance(output, Opts) and opts is None:
        opts = output
        output = None

    syst = deepcopy(syst)
    epsilon_xx, epsilon_yy, epsilon_zz = _as_3d_epsilon(syst)
    off_diagonal = _as_3d_off_diagonal_epsilon(syst, epsilon_xx, epsilon_yy, epsilon_zz)
    has_off_diagonal = any(component is not None for component in off_diagonal.values())
    syst.epsilon_low = _as_real_scalar(syst.epsilon_low, "syst.epsilon_low")
    syst.epsilon_high = _as_real_scalar(syst.epsilon_high, "syst.epsilon_high", allow_none=True)
    if syst.wavelength is None or syst.dx is None or syst.dx <= 0:
        raise ValueError("syst.wavelength and positive syst.dx must be set.")

    opts = _prepare_opts(output, opts)
    if opts.symmetrize_K:
        raise NotImplementedError("opts.symmetrize_K is not supported for 3D mesti2s.")

    two_sided = syst.epsilon_high is not None
    _validate_3d_field_extension_shapes((epsilon_xx.shape, epsilon_yy.shape, epsilon_zz.shape), two_sided)
    pml_layers, nz_extra = _as_pml_list(syst.zPML, two_sided)
    xBC, yBC = _as_3d_boundary_phases(syst, epsilon_xx, epsilon_yy)
    k0dx = (2 * np.pi / syst.wavelength) * syst.dx
    channels = mesti_build_channels(
        epsilon_xx.shape[0],
        epsilon_yy.shape[0],
        xBC,
        epsilon_xx.shape[1],
        epsilon_yy.shape[1],
        yBC,
        k0dx,
        syst.epsilon_low,
        syst.epsilon_high if two_sided else None,
        bool(opts.use_continuous_dispersion),
        opts.n0,
        opts.m0,
    )

    low = _side_low(channels)
    high = _side_high(channels) if isinstance(channels, Channels_two_sided) else None
    input_parsed = _parse_3d_selector(input, low.N_prop, high.N_prop if high is not None else None, "input")
    _ensure_nonempty_selection_3d(input_parsed, "input")
    output_parsed = None
    if output is not None:
        output_parsed = _parse_3d_selector(output, low.N_prop, high.N_prop if high is not None else None, "output")
        _ensure_nonempty_selection_3d(output_parsed, "output")

    modes_low = _surface_modes_3d(channels, low)
    alpha_low = _polarization_coefficients(low)
    modes_high = None
    alpha_high = None
    if high is not None:
        if syst.epsilon_high == syst.epsilon_low:
            modes_high = modes_low
            alpha_high = alpha_low
        else:
            modes_high = _surface_modes_3d(channels, high)
            alpha_high = _polarization_coefficients(high)

    shape_ex = epsilon_xx.shape[:2]
    shape_ey = epsilon_yy.shape[:2]
    B_Ex_low, B_Ey_low = _build_input_surface_3d(modes_low, low, alpha_low, input_parsed, "low", shape_ex, shape_ey)
    B_pos_ex = [_surface_pos_3d(shape_ex[0], shape_ex[1], nz_extra[0] - 1)]
    B_pos_ey = [_surface_pos_3d(shape_ey[0], shape_ey[1], nz_extra[0] - 1)]
    B_data_ex = [B_Ex_low]
    B_data_ey = [B_Ey_low]
    if high is not None:
        B_Ex_high, B_Ey_high = _build_input_surface_3d(
            modes_high, high, alpha_high, input_parsed, "high", shape_ex, shape_ey
        )
        # The high source/projection surface sits one Yee z-index past the
        # original scattering region after low padding.
        high_z = nz_extra[0] + epsilon_xx.shape[2]
        B_pos_ex.append(_surface_pos_3d(shape_ex[0], shape_ex[1], high_z))
        B_pos_ey.append(_surface_pos_3d(shape_ey[0], shape_ey[1], high_z))
        B_data_ex.append(B_Ex_high)
        B_data_ey.append(B_Ey_high)
    B = [
        Source_struct(pos=B_pos_ex, data=B_data_ex),
        Source_struct(pos=B_pos_ey, data=B_data_ey),
        _empty_component_source(),
    ]

    C = None
    if output_parsed is not None:
        C_Ex_low, C_Ey_low = _build_output_surface_3d(
            modes_low, low, alpha_low, output_parsed, "low", shape_ex, shape_ey
        )
        C_pos_ex = [_surface_pos_3d(shape_ex[0], shape_ex[1], nz_extra[0] - 1)]
        C_pos_ey = [_surface_pos_3d(shape_ey[0], shape_ey[1], nz_extra[0] - 1)]
        C_data_ex = [C_Ex_low]
        C_data_ey = [C_Ey_low]
        if high is not None:
            C_Ex_high, C_Ey_high = _build_output_surface_3d(
                modes_high, high, alpha_high, output_parsed, "high", shape_ex, shape_ey
            )
            high_z = nz_extra[0] + epsilon_xx.shape[2]
            C_pos_ex.append(_surface_pos_3d(shape_ex[0], shape_ex[1], high_z))
            C_pos_ey.append(_surface_pos_3d(shape_ey[0], shape_ey[1], high_z))
            C_data_ex.append(C_Ex_high)
            C_data_ey.append(C_Ey_high)
        C = [
            Source_struct(pos=C_pos_ex, data=C_data_ex),
            Source_struct(pos=C_pos_ey, data=C_data_ey),
            _empty_component_source(),
        ]

    padded_syst = _padded_system_3d(
        syst,
        epsilon_xx,
        epsilon_yy,
        epsilon_zz,
        off_diagonal,
        pml_layers,
        nz_extra,
        two_sided,
        xBC,
        yBC,
    )
    result = mesti(padded_syst, B, C=C, opts=opts)

    if output is None:
        Ex, Ey, Ez, info = result
        Ex = (-2j) * Ex
        Ey = (-2j) * Ey
        Ez = (-2j) * Ez
        if input_parsed["use_indices"]:
            prefactor = _input_prefactor_3d(input_parsed, low, high)
            Ex = Ex * prefactor.reshape(1, 1, 1, -1)
            Ey = Ey * prefactor.reshape(1, 1, 1, -1)
            Ez = Ez * prefactor.reshape(1, 1, 1, -1)
        Ex, Ey, Ez = _crop_field_profile_3d(
            Ex,
            Ey,
            Ez,
            (epsilon_xx.shape, epsilon_yy.shape, epsilon_zz.shape),
            nz_extra,
            opts,
            channels,
            input_parsed,
            two_sided,
        )
        info.opts.return_field_profile = True
        info.opts.symmetrize_K = False
        return Ex, Ey, Ez, channels, info

    S, info = result
    S = (-2j) * S
    if input_parsed["use_indices"]:
        S = S * _input_prefactor_3d(input_parsed, low, high).reshape(1, -1)
    if output_parsed["use_indices"]:
        S = _input_prefactor_3d(output_parsed, low, high).reshape(-1, 1) * S
    S = _subtract_direct_terms_3d(S, low, high, input_parsed, output_parsed)
    info.opts.return_field_profile = False
    info.opts.symmetrize_K = False
    return S, channels, info


def _mesti2s_2d_tm(
    syst: Syst,
    input: channel_type | channel_index | wavefront,
    output: channel_type | channel_index | wavefront | Opts | None = None,
    opts: Opts | None = None,
) -> tuple[np.ndarray, Channels_one_sided | Channels_two_sided, Info]:
    """Compute the first supported 2D TM ``mesti2s`` paths.

    Supported paths are 2D TM channel/wavefront inputs from the low side and,
    for two-sided systems, channel/wavefront outputs on the high side.  Channel
    indices in Python are zero-based.
    """

    if isinstance(output, Opts) and opts is None:
        opts = output
        output = None

    if not isinstance(syst, Syst):
        raise TypeError("syst must be a Syst instance.")
    syst = deepcopy(syst)
    epsilon_xx = _as_2d_tm_epsilon(syst)
    syst.epsilon_low = _as_real_scalar(syst.epsilon_low, "syst.epsilon_low")
    syst.epsilon_high = _as_real_scalar(syst.epsilon_high, "syst.epsilon_high", allow_none=True)
    if syst.wavelength is None or syst.dx is None or syst.dx <= 0:
        raise ValueError("syst.wavelength and positive syst.dx must be set.")
    if syst.yBC is None and syst.ky_B is None:
        raise ValueError("syst.yBC or syst.ky_B must be set.")

    two_sided = syst.epsilon_high is not None
    opts = _prepare_opts(output, opts)
    _prefer_mumpspy_apf_for_projected_2d_solve(opts, output)
    symmetrize_requested = _as_optional_bool(opts.symmetrize_K, "opts.symmetrize_K")
    pml_layers, nz_extra = _as_pml_list(syst.zPML, two_sided)

    ny, nz = epsilon_xx.shape
    # Julia converts syst.ky_B to the dimensionless Bloch phase ky_B*Lambda_y
    # before building channels and the padded FDFD system.
    yBC = syst.ky_B * (ny * syst.dx) if syst.ky_B is not None else syst.yBC
    k0dx = (2 * np.pi / syst.wavelength) * syst.dx
    channels = mesti_build_channels(
        ny,
        yBC,
        k0dx,
        syst.epsilon_low,
        syst.epsilon_high if two_sided else None,
        bool(opts.use_continuous_dispersion),
        opts.m0,
    )

    low = _side_low(channels)
    high = _side_high(channels) if isinstance(channels, Channels_two_sided) else None
    input_parsed = _parse_input(input, low.N_prop, high.N_prop if high is not None else None)
    _ensure_nonempty_selection(input_parsed, "input")
    output_parsed = None if output is None else _parse_output(output, low.N_prop, high.N_prop if high is not None else None)
    if output_parsed is not None:
        _ensure_nonempty_selection(output_parsed, "output")

    use_symmetrized_k = False
    sym_output_positions = None
    sym_input_positions = None
    input_for_solve = input_parsed
    if symmetrize_requested:
        if output_parsed is None:
            raise NotImplementedError("opts.symmetrize_K requires scattering-matrix output, not field profiles.")
        if not input_parsed["use_indices"] or not output_parsed["use_indices"]:
            raise NotImplementedError("opts.symmetrize_K is supported only for channel_type/channel_index inputs and outputs.")
        if not _can_symmetrize_2d_transverse_operator(yBC):
            raise NotImplementedError("opts.symmetrize_K requires a symmetric 2D TM transverse operator; nonzero Bloch yBC is not supported.")
        use_symmetrized_k = True
        input_for_solve, sym_output_positions, sym_input_positions = _symmetrized_channel_expansion(
            input_parsed,
            output_parsed,
            low,
            high,
        )
        opts.symmetrize_K = None

    f_prop_low = channels.f_x_m(low.kydx_prop)
    f_prop_high = None
    if high is not None:
        f_prop_high = f_prop_low if syst.epsilon_high == syst.epsilon_low else channels.f_x_m(high.kydx_prop)

    B_pos = [_surface_pos(ny, nz_extra[0] - 1)]
    B_data = [_build_input_block(f_prop_low, low, input_for_solve, "low")]
    if high is not None:
        # Low and high surfaces sit just outside the scattering region after
        # padding: z = nz_extra_low-1 and z = nz_extra_low+nz in zero-based
        # Python indexing, matching Julia's l_low/l_high source planes.
        B_pos.append(_surface_pos(ny, nz_extra[0] + nz))
        B_data.append(_build_input_block(f_prop_high, high, input_for_solve, "high"))
    B = [Source_struct(pos=B_pos, data=B_data)]

    C = None
    if use_symmetrized_k:
        C = "transpose(B)"
    elif output_parsed is not None:
        C_pos = [_surface_pos(ny, nz_extra[0] - 1)]
        C_data = [_build_output_block(f_prop_low, low, output_parsed, "low")]
        if high is not None:
            C_pos.append(_surface_pos(ny, nz_extra[0] + nz))
            C_data.append(_build_output_block(f_prop_high, high, output_parsed, "high"))
        C = [Source_struct(pos=C_pos, data=C_data)]

    padded_syst = _padded_system(syst, epsilon_xx, pml_layers, nz_extra, two_sided, yBC)
    result, info = mesti(padded_syst, B, C=C, opts=opts)
    info.opts.return_field_profile = output is None
    info.opts.symmetrize_K = bool(use_symmetrized_k)

    if output is None:
        Ex = (-2j) * result
        if input_parsed["use_indices"]:
            prefactor = _phase_prefactor(low, input_parsed["low"])
            if high is not None:
                prefactor = np.concatenate([prefactor, _phase_prefactor(high, input_parsed["high"])])
            Ex = Ex * prefactor.reshape(1, 1, -1)
        Ex = _crop_field_profile(Ex, nz, nz_extra, two_sided, opts, channels, low, high, input_parsed)
        return Ex, channels, info

    S = (-2j) * result
    if use_symmetrized_k:
        S = S[np.ix_(sym_output_positions, sym_input_positions)]
    if input_parsed["use_indices"]:
        prefactor = _phase_prefactor(low, input_parsed["low"])
        if high is not None:
            prefactor = np.concatenate([prefactor, _phase_prefactor(high, input_parsed["high"])])
        S = S * prefactor.reshape(1, -1)
    if output_parsed["use_indices"]:
        prefactor = _phase_prefactor(low, output_parsed["low"])
        if high is not None:
            prefactor = np.concatenate([prefactor, _phase_prefactor(high, output_parsed["high"])])
        S = prefactor.reshape(-1, 1) * S
    S = _subtract_direct_terms(S, low, high, input_parsed, output_parsed)
    return S, channels, info


def mesti2s(
    syst: Syst,
    input: channel_type | channel_index | wavefront,
    output: channel_type | channel_index | wavefront | Opts | None = None,
    opts: Opts | None = None,
) -> tuple[Any, ...]:
    """Compute supported ``mesti2s`` scattering or field-profile paths.

    The Python port supports the established 2D TM/scalar wrapper plus the
    fixture-backed 3D vectorial tensor path for s/p polarization channels.
    Channel indices are zero-based in both 2D and 3D Python selectors.
    """

    if isinstance(output, Opts) and opts is None:
        opts = output
        output = None

    if not isinstance(syst, Syst):
        raise TypeError("syst must be a Syst instance.")
    epsilon_xx = np.asarray(syst.epsilon_xx)
    if epsilon_xx.ndim == 2:
        return _mesti2s_2d_tm(syst, input, output, opts)
    if epsilon_xx.ndim == 3:
        return _mesti2s_3d_diagonal(syst, input, output, opts)
    raise NotImplementedError("Only 2D TM and 3D vectorial mesti2s paths are ported so far.")
