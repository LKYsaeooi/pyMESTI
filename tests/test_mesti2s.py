import importlib
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import mesti.cudss_backend as cudss_backend
from mesti import (
    Channels_one_sided,
    Channels_two_sided,
    Opts,
    PML,
    Syst,
    channel_index,
    channel_type,
    mesti2s,
    mesti_build_channels,
    wavefront,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
MESTI2S_3D_FIXTURE = FIXTURE_DIR / "mesti2s_3d_diagonal_periodic.mat"
MESTI2S_2D_APF_DEFAULT_V5_FIXTURE = FIXTURE_DIR / "mesti2s_2d_tm_apf_default_v5.mat"
MESTI2S_2D_SYMMETRIZED_K_V5_FIXTURE = FIXTURE_DIR / "mesti2s_2d_tm_symmetrized_k_v5.mat"
MESTI2S_3D_V5_FIXTURE = FIXTURE_DIR / "mesti2s_3d_diagonal_v5_boundaries.mat"
MESTI2S_3D_NZ_FIXTURE = FIXTURE_DIR / "mesti2s_3d_nz_extension.mat"
MESTI2S_3D_OFFDIAGONAL_V5_FIXTURE = FIXTURE_DIR / "mesti2s_3d_offdiagonal_v5.mat"
MESTI2S_3D_RTOL = 5e-8
MESTI2S_3D_ATOL = 5e-9


def _skip_unless_cudss_available(testcase):
    probe = cudss_backend.probe_environment()
    if not probe.available:
        testcase.skipTest(probe.unavailable_reason or "cuDSS GPU environment is not available")
    if probe.binding_strategy != "nvmath-bindings":
        testcase.skipTest("cuDSS wrapper tests require nvmath.bindings.cudss")


def _load_fixture(path):
    try:
        return {
            key: value
            for key, value in loadmat(path, squeeze_me=False).items()
            if not key.startswith("__")
        }
    except NotImplementedError:
        import h5py

        data = {}
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


def _scalar(data, key):
    value = np.asarray(data[key])
    if value.size != 1:
        raise AssertionError(f"Fixture key {key!r} is not scalar.")
    return value.reshape(-1)[0].item()


def _string(data, key):
    return str(_scalar(data, key))


def _bc_value(data, key):
    value = np.asarray(data[key])
    scalar = value.reshape(-1)[0]
    if np.issubdtype(value.dtype, np.number):
        return scalar.item()
    return str(scalar)


def _vector(data, key, dtype=None):
    return np.asarray(data[key], dtype=dtype).reshape(-1)


def _three_d_system_from_fixture(data, prefix="", two_sided=True):
    high_key = f"{prefix}epsilon_high"
    kwargs = dict(
        epsilon_xx=np.asarray(data[f"{prefix}epsilon_xx"], dtype=np.complex128),
        epsilon_yy=np.asarray(data[f"{prefix}epsilon_yy"], dtype=np.complex128),
        epsilon_zz=np.asarray(data[f"{prefix}epsilon_zz"], dtype=np.complex128),
        epsilon_low=float(_scalar(data, f"{prefix}epsilon_low")),
        epsilon_high=float(_scalar(data, high_key)) if two_sided and high_key in data else None,
        wavelength=float(_scalar(data, f"{prefix}wavelength")),
        dx=float(_scalar(data, f"{prefix}dx")),
        xBC=_bc_value(data, f"{prefix}xBC"),
        yBC=_bc_value(data, f"{prefix}yBC"),
        zPML=[PML(int(_scalar(data, f"{prefix}zPML_npixels")))],
    )
    for name in ("epsilon_xy", "epsilon_xz", "epsilon_yx", "epsilon_yz", "epsilon_zx", "epsilon_zy"):
        key = f"{prefix}{name}"
        if key in data:
            kwargs[name] = np.asarray(data[key], dtype=np.complex128)
    return Syst(**kwargs)


def _two_d_system_from_fixture(data):
    return Syst(
        epsilon_xx=np.asarray(data["epsilon_xx"], dtype=np.complex128),
        epsilon_low=float(_scalar(data, "epsilon_low")),
        epsilon_high=float(_scalar(data, "epsilon_high")) if "epsilon_high" in data else None,
        wavelength=float(_scalar(data, "wavelength")),
        dx=float(_scalar(data, "dx")),
        yBC=_bc_value(data, "yBC"),
        zPML=[PML(int(_scalar(data, "zPML_npixels")))],
    )


def _assert_3d_side_metadata(testcase, side, data, prefix):
    testcase.assertEqual(side.N_prop, int(_scalar(data, f"{prefix}_N_prop")))
    testcase.assertEqual(
        np.asarray(side.ind_prop).tolist(),
        _vector(data, f"{prefix}_ind_prop_zero_based", dtype=int).tolist(),
    )
    np.testing.assert_allclose(
        side.kxdx_prop,
        _vector(data, f"{prefix}_kxdx_prop", dtype=np.complex128),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        side.kydx_prop,
        _vector(data, f"{prefix}_kydx_prop", dtype=np.complex128),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        side.kzdx_prop,
        _vector(data, f"{prefix}_kzdx_prop", dtype=np.complex128),
        rtol=1e-12,
        atol=1e-12,
    )


def _interface_syst(n_low=1.35, n_high=1.8):
    return Syst(
        epsilon_xx=np.ones((1, 0), dtype=np.complex128),
        epsilon_low=n_low**2,
        epsilon_high=n_high**2,
        wavelength=1.0,
        dx=0.1,
        yBC="periodic",
        zPML=[PML(20)],
    )


def _patterned_real_3d(shape, base):
    ix, iy, iz = np.indices(shape, dtype=float)
    return (base + 0.018 * ix + 0.011 * iy + 0.007 * iz + 0.003 * np.sin(ix + iy + iz)).astype(
        np.complex128
    )


def _patterned_complex_3d(shape, scale):
    ix, iy, iz = np.indices(shape, dtype=float)
    return (
        scale * (0.27 + 0.06 * ix - 0.035 * iy + 0.025 * iz)
        + 1j * scale * (0.16 - 0.045 * ix + 0.018 * iy + 0.032 * iz)
    ).astype(np.complex128)


def _lossless_3d_unitarity_system(include_off_diagonal=False):
    syst = Syst(
        epsilon_xx=_patterned_real_3d((2, 3, 1), 1.10),
        epsilon_yy=_patterned_real_3d((2, 3, 1), 1.22),
        epsilon_zz=_patterned_real_3d((2, 3, 2), 1.34),
        epsilon_low=1.0,
        epsilon_high=1.0,
        wavelength=5.0,
        dx=1.0,
        xBC="periodic",
        yBC="periodic",
        zPML=[PML(16)],
    )
    if include_off_diagonal:
        epsilon_xy = _patterned_complex_3d((2, 3, 1), 0.030)
        epsilon_xz = _patterned_complex_3d((2, 3, 1), -0.025)
        epsilon_yz = _patterned_complex_3d((2, 3, 1), 0.020)
        syst.epsilon_xy = epsilon_xy
        syst.epsilon_yx = np.conjugate(epsilon_xy)
        syst.epsilon_xz = epsilon_xz
        syst.epsilon_zx = np.conjugate(epsilon_xz)
        syst.epsilon_yz = epsilon_yz
        syst.epsilon_zy = np.conjugate(epsilon_yz)
    return syst


def _unitarity_residual(S):
    return np.max(np.abs(S.conjugate().T @ S - np.eye(S.shape[1], dtype=np.complex128)))


def _small_2d_syst(two_sided=True):
    return Syst(
        epsilon_xx=np.array(
            [
                [1.00 + 0.00j, 1.04 + 0.02j],
                [1.03 + 0.01j, 1.07 + 0.00j],
                [0.98 + 0.00j, 1.02 + 0.01j],
                [1.01 + 0.02j, 1.06 + 0.01j],
            ],
            dtype=np.complex128,
        ),
        epsilon_low=1.21,
        epsilon_high=1.44 if two_sided else None,
        wavelength=float(2 * np.pi / 1.4),
        dx=1.0,
        yBC="periodic",
        zPML=[PML(3)],
    )


def _cutoff_syst(*, epsilon_low=1.0, epsilon_high=1.0):
    return Syst(
        epsilon_xx=np.ones((1, 0), dtype=np.complex128),
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        wavelength=1.0,
        dx=0.1,
        yBC="periodic",
        zPML=[PML(3)],
    )


class Mesti2STest(unittest.TestCase):
    def test_cudss_low_to_high_scattering_matches_scipy_when_available(self):
        _skip_unless_cudss_available(self)
        syst = _interface_syst()
        inp = channel_type(side="low")
        out = channel_type(side="high")

        S_cudss, channels_cudss, info = mesti2s(syst, inp, out, Opts(solver="cudss", verbal=False))
        S_scipy, channels_scipy, _ = mesti2s(syst, inp, out, Opts(solver="scipy", verbal=False))

        np.testing.assert_allclose(S_cudss, S_scipy, rtol=1e-10, atol=1e-10)
        self.assertEqual(channels_cudss.low.N_prop, channels_scipy.low.N_prop)
        self.assertEqual(channels_cudss.high.N_prop, channels_scipy.high.N_prop)
        self.assertEqual(info.opts.solver, "cudss")
        self.assertFalse(info.opts.return_field_profile)

    def test_prefactor_option_is_rejected_for_mesti2s(self):
        syst = _small_2d_syst(two_sided=True)

        with self.assertRaisesRegex(ValueError, "prefactor"):
            mesti2s(
                syst,
                channel_type(side="low"),
                channel_type(side="high"),
                Opts(verbal=False, prefactor=2.0),
            )

    def test_channel_type_low_to_high_matches_1d_interface_formula(self):
        syst = _interface_syst()
        inp = channel_type(side="low")
        out = channel_type(side="high")

        S, channels, info = mesti2s(syst, inp, out, Opts(verbal=False))

        kz_low = channels.low.kzdx_prop[0]
        kz_high = channels.high.kzdx_prop[0]
        expected_t = (
            np.sqrt(np.sin(kz_high))
            / np.sqrt(np.sin(kz_low))
            * (np.exp(1j * kz_low * 1.5) - np.exp(-1j * kz_low * 0.5))
            / (np.exp(1j * kz_high * 0.5) * np.exp(1j * kz_low) - np.exp(-1j * kz_high * 0.5))
        )

        self.assertEqual(S.shape, (1, 1))
        np.testing.assert_allclose(S[0, 0], expected_t, atol=1e-4)
        self.assertFalse(info.opts.return_field_profile)

    def test_wavefront_identity_matches_channel_field_path(self):
        syst = Syst(
            epsilon_xx=np.ones((3, 2), dtype=np.complex128),
            epsilon_low=1.0,
            epsilon_high=1.0,
            wavelength=2 * np.pi,
            dx=1.0,
            yBC="periodic",
            zPML=[PML(4)],
        )

        channels_field, _, _ = (None, None, None)
        channel_ex, channels_field, _ = mesti2s(
            syst,
            channel_type(side="low"),
            Opts(verbal=False),
        )
        wave_ex, channels_wave, info = mesti2s(
            syst,
            wavefront(v_low=np.eye(channels_field.low.N_prop, dtype=np.complex128)),
            Opts(verbal=False),
        )

        self.assertEqual(channel_ex.shape, (3, 2, channels_field.low.N_prop))
        self.assertEqual(wave_ex.shape, channel_ex.shape)
        self.assertEqual(channels_wave.low.N_prop, channels_field.low.N_prop)
        self.assertTrue(info.opts.return_field_profile)
        np.testing.assert_allclose(wave_ex, channel_ex, atol=1e-10)

    def test_two_sided_field_profile_can_include_homogeneous_side_pixels(self):
        syst = _small_2d_syst(two_sided=True)
        v_low = np.array([[1.0], [0.25 + 0.5j], [-0.5j]], dtype=np.complex128)

        core_ex, channels_core, _ = mesti2s(
            syst,
            wavefront(v_low=v_low),
            Opts(verbal=False),
        )
        extended_ex, channels_extended, info = mesti2s(
            syst,
            wavefront(v_low=v_low),
            Opts(verbal=False, nz_low=3, nz_high=2),
        )

        self.assertEqual(channels_extended.low.N_prop, channels_core.low.N_prop)
        self.assertEqual(extended_ex.shape, (4, 7, 1))
        self.assertTrue(info.opts.return_field_profile)
        np.testing.assert_allclose(extended_ex[:, 3:5, :], core_ex, atol=1e-10)
        self.assertGreater(np.linalg.norm(extended_ex[:, :3, :]), 0)
        self.assertGreater(np.linalg.norm(extended_ex[:, 5:, :]), 0)

    def test_one_sided_field_profile_extends_low_and_zero_pads_high(self):
        syst = _small_2d_syst(two_sided=False)

        core_ex, channels_core, _ = mesti2s(
            syst,
            channel_type(side="low"),
            Opts(verbal=False),
        )
        extended_ex, channels_extended, _ = mesti2s(
            syst,
            channel_type(side="low"),
            Opts(verbal=False, nz_low=2, nz_high=2),
        )

        self.assertEqual(channels_extended.N_prop, channels_core.N_prop)
        self.assertEqual(extended_ex.shape, (4, 6, channels_core.N_prop))
        np.testing.assert_allclose(extended_ex[:, 2:4, :], core_ex, atol=1e-10)
        self.assertGreater(np.linalg.norm(extended_ex[:, :2, :]), 0)
        np.testing.assert_allclose(extended_ex[:, -2:, :], 0, atol=1e-12)

    def test_one_sided_low_reflection_matches_julia_reference(self):
        syst = _small_2d_syst(two_sided=False)
        expected = np.array(
            [
                [
                    1.4846229400788293 + 0.11738000601280557j,
                    0.013932435830653862 + 0.00044331501329042053j,
                    -0.044832681055242594 + 0.07083090264509641j,
                ],
                [
                    -0.0029830614029702205 + 0.001475654888624122j,
                    0.25374124319038283 - 0.7049468659569504j,
                    0.01393243583065386 + 0.00044331501329042313j,
                ],
                [
                    -0.04378342126263047 + 0.0706435661678825j,
                    -0.002983061402970221 + 0.001475654888624122j,
                    1.4846229400788293 + 0.11738000601280557j,
                ],
            ],
            dtype=np.complex128,
        )

        S, channels, info = mesti2s(
            syst,
            channel_type(side="low"),
            channel_type(side="low"),
            Opts(verbal=False),
        )

        self.assertEqual(channels.N_prop, 3)
        self.assertEqual(S.shape, (3, 3))
        self.assertFalse(info.opts.return_field_profile)
        np.testing.assert_allclose(S, expected, rtol=5e-5, atol=2e-5)

    def test_one_sided_channel_index_reflection_matches_channel_type_submatrix(self):
        syst = _small_2d_syst(two_sided=False)
        selected = np.array([0, 2], dtype=int)

        full, _, _ = mesti2s(
            syst,
            channel_type(side="low"),
            channel_type(side="low"),
            Opts(verbal=False),
        )
        subset, channels, info = mesti2s(
            syst,
            channel_index(ind_low=selected),
            channel_index(ind_low=selected),
            Opts(verbal=False),
        )

        self.assertEqual(channels.N_prop, 3)
        self.assertEqual(subset.shape, (2, 2))
        self.assertFalse(info.opts.return_field_profile)
        np.testing.assert_allclose(subset, full[np.ix_(selected, selected)], atol=1e-10)

    def test_field_profile_side_pixel_counts_must_be_integers(self):
        syst = _small_2d_syst(two_sided=True)

        with self.assertRaisesRegex(ValueError, "opts.nz_low"):
            mesti2s(
                syst,
                channel_type(side="low"),
                Opts(verbal=False, nz_low=1.5),
            )

    def test_2d_mesti2s_scipy_matches_julia_default_apf_fixture(self):
        fixture = _load_fixture(MESTI2S_2D_APF_DEFAULT_V5_FIXTURE)
        syst = _two_d_system_from_fixture(fixture)

        S, channels, info = mesti2s(
            syst,
            channel_type(side="low"),
            channel_type(side="high"),
            Opts(solver="scipy", verbal=False),
        )

        self.assertIsInstance(channels, Channels_two_sided)
        self.assertEqual(channels.low.N_prop, int(_scalar(fixture, "low_N_prop")))
        self.assertEqual(channels.high.N_prop, int(_scalar(fixture, "high_N_prop")))
        self.assertEqual(info.opts.method, "factorize_and_solve")
        self.assertFalse(info.opts.return_field_profile)
        np.testing.assert_allclose(S, np.asarray(fixture["S_default_apf"], dtype=np.complex128), rtol=5e-8, atol=5e-8)
        np.testing.assert_allclose(
            S,
            np.asarray(fixture["S_factorize_and_solve"], dtype=np.complex128),
            rtol=5e-8,
            atol=5e-8,
        )
        np.testing.assert_allclose(
            np.linalg.svd(S, compute_uv=False),
            _vector(fixture, "S_default_singular_values", dtype=float),
            rtol=5e-8,
            atol=5e-8,
        )

    @unittest.skipUnless(importlib.util.find_spec("mumpspy") is not None, "mumpspy is not installed")
    def test_2d_mesti2s_mumpspy_projected_default_uses_apf(self):
        fixture = _load_fixture(MESTI2S_2D_APF_DEFAULT_V5_FIXTURE)
        syst = _two_d_system_from_fixture(fixture)

        S, _, info = mesti2s(
            syst,
            channel_type(side="low"),
            channel_type(side="high"),
            Opts(solver="mumpspy", verbal=False),
        )

        self.assertEqual(info.opts.solver, "mumpspy")
        self.assertEqual(info.opts.method, "APF")
        self.assertFalse(info.opts.return_field_profile)
        np.testing.assert_allclose(S, np.asarray(fixture["S_default_apf"], dtype=np.complex128), rtol=5e-8, atol=5e-8)

    def test_2d_mesti2s_symmetrized_k_matches_julia_fixture(self):
        fixture = _load_fixture(MESTI2S_2D_SYMMETRIZED_K_V5_FIXTURE)
        syst = _two_d_system_from_fixture(fixture)
        inp = channel_index(
            ind_low=_vector(fixture, "input_ind_low_zero_based", dtype=int),
            ind_high=_vector(fixture, "input_ind_high_zero_based", dtype=int),
        )
        out = channel_index(
            ind_low=_vector(fixture, "output_ind_low_zero_based", dtype=int),
            ind_high=_vector(fixture, "output_ind_high_zero_based", dtype=int),
        )

        S, channels, info = mesti2s(
            syst,
            inp,
            out,
            Opts(solver="scipy", verbal=False, symmetrize_K=True),
        )

        self.assertIsInstance(channels, Channels_two_sided)
        self.assertTrue(info.opts.symmetrize_K)
        self.assertFalse(info.opts.return_field_profile)
        self.assertEqual(S.shape, np.asarray(fixture["S_sym"]).shape)
        np.testing.assert_allclose(S, np.asarray(fixture["S_sym"], dtype=np.complex128), rtol=5e-8, atol=5e-8)
        np.testing.assert_allclose(S, np.asarray(fixture["S_unsym"], dtype=np.complex128), rtol=5e-8, atol=5e-8)
        np.testing.assert_allclose(
            np.linalg.svd(S, compute_uv=False),
            _vector(fixture, "S_sym_singular_values", dtype=float),
            rtol=5e-8,
            atol=5e-8,
        )

    def test_2d_mesti2s_symmetrized_k_channel_expansion_matches_julia_fixture(self):
        fixture = _load_fixture(MESTI2S_2D_SYMMETRIZED_K_V5_FIXTURE)
        syst = _two_d_system_from_fixture(fixture)
        channels = mesti_build_channels(syst)
        mesti2s_module = importlib.import_module("mesti.mesti2s")
        input_parsed = {
            "use_indices": True,
            "low": _vector(fixture, "input_ind_low_zero_based", dtype=int),
            "high": _vector(fixture, "input_ind_high_zero_based", dtype=int),
        }
        output_parsed = {
            "use_indices": True,
            "low": _vector(fixture, "output_ind_low_zero_based", dtype=int),
            "high": _vector(fixture, "output_ind_high_zero_based", dtype=int),
        }

        solve_parsed, output_positions, input_positions = mesti2s_module._symmetrized_channel_expansion(
            input_parsed,
            output_parsed,
            channels.low,
            channels.high,
        )

        np.testing.assert_array_equal(channels.low.ind_prop_conj, _vector(fixture, "low_ind_prop_conj_zero_based", dtype=int))
        np.testing.assert_array_equal(channels.high.ind_prop_conj, _vector(fixture, "high_ind_prop_conj_zero_based", dtype=int))
        np.testing.assert_array_equal(solve_parsed["low"], _vector(fixture, "low_expanded_zero_based", dtype=int))
        np.testing.assert_array_equal(solve_parsed["high"], _vector(fixture, "high_expanded_zero_based", dtype=int))
        expected_input_positions = np.concatenate(
            [
                _vector(fixture, "low_input_positions_zero_based", dtype=int),
                len(solve_parsed["low"]) + _vector(fixture, "high_input_positions_zero_based", dtype=int),
            ]
        )
        expected_output_positions = np.concatenate(
            [
                _vector(fixture, "low_output_positions_zero_based", dtype=int),
                len(solve_parsed["low"]) + _vector(fixture, "high_output_positions_zero_based", dtype=int),
            ]
        )
        np.testing.assert_array_equal(input_positions, expected_input_positions)
        np.testing.assert_array_equal(output_positions, expected_output_positions)

    def test_2d_mesti2s_symmetrize_k_unsupported_paths_are_explicit(self):
        syst = _small_2d_syst(two_sided=True)

        with self.assertRaisesRegex(NotImplementedError, "field profiles"):
            mesti2s(
                syst,
                channel_type(side="low"),
                Opts(solver="scipy", verbal=False, symmetrize_K=True),
            )

        with self.assertRaisesRegex(NotImplementedError, "channel_type/channel_index"):
            mesti2s(
                syst,
                wavefront(v_low=np.ones((3, 1), dtype=np.complex128)),
                channel_type(side="high"),
                Opts(solver="scipy", verbal=False, symmetrize_K=True),
            )

        bloch = _small_2d_syst(two_sided=True)
        bloch.yBC = "Bloch"
        bloch.ky_B = 0.2
        with self.assertRaisesRegex(NotImplementedError, "nonzero Bloch"):
            mesti2s(
                bloch,
                channel_type(side="low"),
                channel_type(side="high"),
                Opts(solver="scipy", verbal=False, symmetrize_K=True),
            )

        with self.assertRaisesRegex(RuntimeError, "APF"):
            mesti2s(
                syst,
                channel_type(side="low"),
                channel_type(side="high"),
                Opts(solver="scipy", method="APF", verbal=False, symmetrize_K=True),
            )

    def test_channel_type_input_must_select_at_least_one_propagating_channel(self):
        syst = _cutoff_syst(epsilon_low=0.0, epsilon_high=1.0)

        with self.assertRaisesRegex(ValueError, "input selects no propagating channels"):
            mesti2s(
                syst,
                channel_type(side="low"),
                channel_type(side="high"),
                Opts(verbal=False),
            )

    def test_channel_type_output_must_select_at_least_one_propagating_channel(self):
        syst = _cutoff_syst(epsilon_low=1.0, epsilon_high=0.0)

        with self.assertRaisesRegex(ValueError, "output selects no propagating channels"):
            mesti2s(
                syst,
                channel_type(side="low"),
                channel_type(side="high"),
                Opts(verbal=False),
            )

    def test_channel_type_both_allows_empty_low_side_when_high_side_propagates(self):
        syst = _cutoff_syst(epsilon_low=1000.0, epsilon_high=1.0)

        S, channels, info = mesti2s(
            syst,
            channel_type(side="both"),
            channel_type(side="both"),
            Opts(verbal=False),
        )

        self.assertEqual(channels.low.N_prop, 0)
        self.assertEqual(channels.high.N_prop, 1)
        self.assertEqual(S.shape, (1, 1))
        self.assertFalse(info.opts.return_field_profile)
        np.testing.assert_allclose(
            S[0, 0],
            -0.7955531301697794 - 0.5532656723010401j,
            rtol=5e-5,
            atol=2e-6,
        )

    def test_channel_type_both_allows_empty_high_side_when_low_side_propagates(self):
        syst = _cutoff_syst(epsilon_low=1.0, epsilon_high=1000.0)

        S, channels, info = mesti2s(
            syst,
            channel_type(side="both"),
            channel_type(side="both"),
            Opts(verbal=False),
        )

        self.assertEqual(channels.low.N_prop, 1)
        self.assertEqual(channels.high.N_prop, 0)
        self.assertEqual(S.shape, (1, 1))
        self.assertFalse(info.opts.return_field_profile)
        np.testing.assert_allclose(
            S[0, 0],
            -0.7955533144939027 - 0.5532658011022724j,
            rtol=5e-5,
            atol=2e-6,
        )

    def test_lossless_2d_scattering_matrix_is_unitary_to_pml_tolerance(self):
        syst = Syst(
            epsilon_xx=np.array(
                [
                    [1.02, 1.12],
                    [1.08, 0.97],
                    [1.11, 1.04],
                    [0.99, 1.06],
                ],
                dtype=np.complex128,
            ),
            epsilon_low=1.0,
            epsilon_high=1.0,
            wavelength=5.0,
            dx=1.0,
            yBC="periodic",
            zPML=[PML(16)],
        )

        S, channels, info = mesti2s(
            syst,
            channel_type(side="both"),
            channel_type(side="both"),
            Opts(verbal=False),
        )

        # For a real permittivity profile, the open-channel S matrix is unitary;
        # the finite PML truncation leaves a small residual.
        self.assertEqual(S.shape[0], S.shape[1])
        self.assertGreater(channels.low.N_prop, 0)
        self.assertGreater(channels.high.N_prop, 0)
        self.assertFalse(info.opts.return_field_profile)
        residual = S.conjugate().T @ S - np.eye(S.shape[1])
        self.assertLessEqual(np.max(np.abs(residual)), 1e-4)

    def test_3d_channel_metadata_matches_julia_fixture(self):
        fixture = _load_fixture(MESTI2S_3D_FIXTURE)
        syst = _three_d_system_from_fixture(fixture)

        channels = mesti_build_channels(syst)

        self.assertIsInstance(channels, Channels_two_sided)
        np.testing.assert_allclose(channels.kxdx_all, _vector(fixture, "kxdx_all", dtype=float))
        np.testing.assert_allclose(channels.kydx_all, _vector(fixture, "kydx_all", dtype=float))
        _assert_3d_side_metadata(self, channels.low, fixture, "low")
        _assert_3d_side_metadata(self, channels.high, fixture, "high")

    def test_3d_two_sided_s_p_scattering_matches_julia_fixture(self):
        fixture = _load_fixture(MESTI2S_3D_FIXTURE)
        syst = _three_d_system_from_fixture(fixture)

        S, channels, info = mesti2s(
            syst,
            channel_type(side="both", polarization="both"),
            channel_type(side="both", polarization="both"),
            Opts(solver="scipy", verbal=False),
        )

        self.assertIsInstance(channels, Channels_two_sided)
        self.assertFalse(info.opts.return_field_profile)
        self.assertEqual(S.shape, np.asarray(fixture["S_both"]).shape)
        np.testing.assert_allclose(S, np.asarray(fixture["S_both"], dtype=np.complex128), rtol=MESTI2S_3D_RTOL, atol=MESTI2S_3D_ATOL)
        np.testing.assert_allclose(
            np.linalg.svd(S, compute_uv=False),
            _vector(fixture, "S_both_singular_values", dtype=float),
            rtol=MESTI2S_3D_RTOL,
            atol=MESTI2S_3D_ATOL,
        )

    def test_3d_zero_based_channel_subselect_matches_julia_fixture(self):
        fixture = _load_fixture(MESTI2S_3D_FIXTURE)
        syst = _three_d_system_from_fixture(fixture)

        S, channels, info = mesti2s(
            syst,
            channel_index(
                ind_low_s=_vector(fixture, "subset_in_low_s_zero_based", dtype=int),
                ind_low_p=_vector(fixture, "subset_in_low_p_zero_based", dtype=int),
                ind_high_s=_vector(fixture, "subset_in_high_s_zero_based", dtype=int),
                ind_high_p=_vector(fixture, "subset_in_high_p_zero_based", dtype=int),
            ),
            channel_index(
                ind_low_p=_vector(fixture, "subset_out_low_p_zero_based", dtype=int),
                ind_high_s=_vector(fixture, "subset_out_high_s_zero_based", dtype=int),
                ind_high_p=_vector(fixture, "subset_out_high_p_zero_based", dtype=int),
            ),
            Opts(solver="scipy", verbal=False),
        )

        self.assertIsInstance(channels, Channels_two_sided)
        self.assertFalse(info.opts.return_field_profile)
        np.testing.assert_allclose(S, np.asarray(fixture["S_subset"], dtype=np.complex128), rtol=MESTI2S_3D_RTOL, atol=MESTI2S_3D_ATOL)

    def test_3d_mixed_wavefront_default_field_profile_matches_julia_fixture(self):
        fixture = _load_fixture(MESTI2S_3D_FIXTURE)
        syst = _three_d_system_from_fixture(fixture)

        Ex, Ey, Ez, channels, info = mesti2s(
            syst,
            wavefront(
                v_low_s=np.asarray(fixture["v_low_s"], dtype=np.complex128),
                v_low_p=np.asarray(fixture["v_low_p"], dtype=np.complex128),
                v_high_p=np.asarray(fixture["v_high_p"], dtype=np.complex128),
            ),
            Opts(solver="scipy", verbal=False),
        )

        self.assertIsInstance(channels, Channels_two_sided)
        self.assertTrue(info.opts.return_field_profile)
        self.assertEqual(Ex.shape, np.asarray(fixture["field_Ex"]).shape)
        self.assertEqual(Ey.shape, np.asarray(fixture["field_Ey"]).shape)
        self.assertEqual(Ez.shape, np.asarray(fixture["field_Ez"]).shape)
        np.testing.assert_allclose(Ex, np.asarray(fixture["field_Ex"], dtype=np.complex128), rtol=MESTI2S_3D_RTOL, atol=MESTI2S_3D_ATOL)
        np.testing.assert_allclose(Ey, np.asarray(fixture["field_Ey"], dtype=np.complex128), rtol=MESTI2S_3D_RTOL, atol=MESTI2S_3D_ATOL)
        np.testing.assert_allclose(Ez, np.asarray(fixture["field_Ez"], dtype=np.complex128), rtol=MESTI2S_3D_RTOL, atol=MESTI2S_3D_ATOL)

    def test_3d_one_sided_low_reflection_matches_julia_manual_fixture(self):
        fixture = _load_fixture(MESTI2S_3D_FIXTURE)
        syst = _three_d_system_from_fixture(fixture, prefix="one_", two_sided=False)

        S, channels, info = mesti2s(
            syst,
            channel_type(side="low", polarization="both"),
            channel_type(side="low", polarization="both"),
            Opts(solver="scipy", verbal=False),
        )

        self.assertIsInstance(channels, Channels_one_sided)
        _assert_3d_side_metadata(self, channels, fixture, "one_low")
        self.assertFalse(info.opts.return_field_profile)
        np.testing.assert_allclose(S, np.asarray(fixture["S_one_low"], dtype=np.complex128), rtol=MESTI2S_3D_RTOL, atol=MESTI2S_3D_ATOL)

    def test_3d_v5_boundary_pml_and_bloch_scattering_matches_julia(self):
        fixture = _load_fixture(MESTI2S_3D_V5_FIXTURE)

        for prefix in ("pml", "bloch", "mixed_bc"):
            with self.subTest(prefix=prefix):
                syst = _three_d_system_from_fixture(fixture, prefix=f"{prefix}_")
                S, channels, info = mesti2s(
                    syst,
                    channel_type(side="both", polarization="both"),
                    channel_type(side="both", polarization="both"),
                    Opts(solver="scipy", verbal=False),
                )

                self.assertIsInstance(channels, Channels_two_sided)
                self.assertFalse(info.opts.return_field_profile)
                _assert_3d_side_metadata(self, channels.low, fixture, f"{prefix}_low")
                _assert_3d_side_metadata(self, channels.high, fixture, f"{prefix}_high")
                np.testing.assert_allclose(channels.kxdx_all, _vector(fixture, f"{prefix}_kxdx_all", dtype=float))
                np.testing.assert_allclose(channels.kydx_all, _vector(fixture, f"{prefix}_kydx_all", dtype=float))
                self.assertEqual(S.shape, np.asarray(fixture[f"{prefix}_S_both"]).shape)
                np.testing.assert_allclose(
                    S,
                    np.asarray(fixture[f"{prefix}_S_both"], dtype=np.complex128),
                    rtol=MESTI2S_3D_RTOL,
                    atol=MESTI2S_3D_ATOL,
                )
                np.testing.assert_allclose(
                    np.linalg.svd(S, compute_uv=False),
                    _vector(fixture, f"{prefix}_S_both_singular_values", dtype=float),
                    rtol=MESTI2S_3D_RTOL,
                    atol=MESTI2S_3D_ATOL,
                )

    def test_3d_offdiagonal_two_sided_s_p_scattering_matches_julia(self):
        fixture = _load_fixture(MESTI2S_3D_OFFDIAGONAL_V5_FIXTURE)
        syst = _three_d_system_from_fixture(fixture, prefix="hermitian_")

        S, channels, info = mesti2s(
            syst,
            channel_type(side="both", polarization="both"),
            channel_type(side="both", polarization="both"),
            Opts(solver="scipy", verbal=False),
        )

        self.assertIsInstance(channels, Channels_two_sided)
        self.assertFalse(info.opts.return_field_profile)
        _assert_3d_side_metadata(self, channels.low, fixture, "hermitian_low")
        _assert_3d_side_metadata(self, channels.high, fixture, "hermitian_high")
        np.testing.assert_allclose(channels.kxdx_all, _vector(fixture, "hermitian_kxdx_all", dtype=float))
        np.testing.assert_allclose(channels.kydx_all, _vector(fixture, "hermitian_kydx_all", dtype=float))
        self.assertEqual(S.shape, np.asarray(fixture["hermitian_S_both"]).shape)
        np.testing.assert_allclose(
            S,
            np.asarray(fixture["hermitian_S_both"], dtype=np.complex128),
            rtol=MESTI2S_3D_RTOL,
            atol=MESTI2S_3D_ATOL,
        )
        np.testing.assert_allclose(
            np.linalg.svd(S, compute_uv=False),
            _vector(fixture, "hermitian_S_both_singular_values", dtype=float),
            rtol=MESTI2S_3D_RTOL,
            atol=MESTI2S_3D_ATOL,
        )
        self.assertLessEqual(_unitarity_residual(S), 1e-3)

    def test_3d_offdiagonal_one_sided_low_reflection_matches_julia_manual_fixture(self):
        fixture = _load_fixture(MESTI2S_3D_OFFDIAGONAL_V5_FIXTURE)
        syst = _three_d_system_from_fixture(fixture, prefix="one_", two_sided=False)

        S, channels, info = mesti2s(
            syst,
            channel_type(side="low", polarization="both"),
            channel_type(side="low", polarization="both"),
            Opts(solver="scipy", verbal=False),
        )

        self.assertIsInstance(channels, Channels_one_sided)
        self.assertFalse(info.opts.return_field_profile)
        _assert_3d_side_metadata(self, channels, fixture, "one_low")
        np.testing.assert_allclose(channels.kxdx_all, _vector(fixture, "one_kxdx_all", dtype=float))
        np.testing.assert_allclose(channels.kydx_all, _vector(fixture, "one_kydx_all", dtype=float))
        self.assertEqual(S.shape, np.asarray(fixture["one_S_low"]).shape)
        np.testing.assert_allclose(
            S,
            np.asarray(fixture["one_S_low"], dtype=np.complex128),
            rtol=MESTI2S_3D_RTOL,
            atol=MESTI2S_3D_ATOL,
        )
        np.testing.assert_allclose(
            np.linalg.svd(S, compute_uv=False),
            _vector(fixture, "one_S_low_singular_values", dtype=float),
            rtol=MESTI2S_3D_RTOL,
            atol=MESTI2S_3D_ATOL,
        )

    def test_lossless_3d_scattering_matrix_is_unitary_to_pml_tolerance(self):
        for include_off_diagonal in (False, True):
            with self.subTest(include_off_diagonal=include_off_diagonal):
                syst = _lossless_3d_unitarity_system(include_off_diagonal)
                S, channels, info = mesti2s(
                    syst,
                    channel_type(side="both", polarization="both"),
                    channel_type(side="both", polarization="both"),
                    Opts(solver="scipy", verbal=False),
                )

                self.assertIsInstance(channels, Channels_two_sided)
                self.assertFalse(info.opts.return_field_profile)
                self.assertEqual(S.shape[0], S.shape[1])
                self.assertGreater(channels.low.N_prop, 0)
                self.assertGreater(channels.high.N_prop, 0)
                residual = _unitarity_residual(S)
                singular_value_deviation = np.max(np.abs(np.linalg.svd(S, compute_uv=False) - 1))
                self.assertLessEqual(residual, 1e-3)
                self.assertLessEqual(singular_value_deviation, 1e-3)

    def test_3d_nz_extension_channel_type_matches_julia_corrected_fixture(self):
        fixture = _load_fixture(MESTI2S_3D_NZ_FIXTURE)
        syst = _three_d_system_from_fixture(fixture)

        Ex, Ey, Ez, channels, info = mesti2s(
            syst,
            channel_type(side="both", polarization="both"),
            Opts(
                solver="scipy",
                verbal=False,
                nz_low=int(_scalar(fixture, "indexed_nz_low")),
                nz_high=int(_scalar(fixture, "indexed_nz_high")),
            ),
        )

        self.assertIsInstance(channels, Channels_two_sided)
        self.assertTrue(info.opts.return_field_profile)
        self.assertEqual(Ex.shape, np.asarray(fixture["indexed_field_Ex"]).shape)
        self.assertEqual(Ey.shape, np.asarray(fixture["indexed_field_Ey"]).shape)
        self.assertEqual(Ez.shape, np.asarray(fixture["indexed_field_Ez"]).shape)
        np.testing.assert_allclose(
            Ex,
            np.asarray(fixture["indexed_field_Ex"], dtype=np.complex128),
            rtol=MESTI2S_3D_RTOL,
            atol=MESTI2S_3D_ATOL,
        )
        np.testing.assert_allclose(
            Ey,
            np.asarray(fixture["indexed_field_Ey"], dtype=np.complex128),
            rtol=MESTI2S_3D_RTOL,
            atol=MESTI2S_3D_ATOL,
        )
        np.testing.assert_allclose(
            Ez,
            np.asarray(fixture["indexed_field_Ez"], dtype=np.complex128),
            rtol=MESTI2S_3D_RTOL,
            atol=MESTI2S_3D_ATOL,
        )

    def test_3d_nz_extension_wavefront_matches_julia_corrected_fixture(self):
        fixture = _load_fixture(MESTI2S_3D_NZ_FIXTURE)
        syst = _three_d_system_from_fixture(fixture)

        Ex, Ey, Ez, channels, info = mesti2s(
            syst,
            wavefront(
                v_low_s=np.asarray(fixture["v_low_s"], dtype=np.complex128),
                v_low_p=np.asarray(fixture["v_low_p"], dtype=np.complex128),
                v_high_s=np.asarray(fixture["v_high_s"], dtype=np.complex128),
                v_high_p=np.asarray(fixture["v_high_p"], dtype=np.complex128),
            ),
            Opts(
                solver="scipy",
                verbal=False,
                nz_low=int(_scalar(fixture, "wavefront_nz_low")),
                nz_high=int(_scalar(fixture, "wavefront_nz_high")),
            ),
        )

        self.assertIsInstance(channels, Channels_two_sided)
        self.assertTrue(info.opts.return_field_profile)
        np.testing.assert_allclose(
            Ex,
            np.asarray(fixture["wavefront_field_Ex"], dtype=np.complex128),
            rtol=MESTI2S_3D_RTOL,
            atol=MESTI2S_3D_ATOL,
        )
        np.testing.assert_allclose(
            Ey,
            np.asarray(fixture["wavefront_field_Ey"], dtype=np.complex128),
            rtol=MESTI2S_3D_RTOL,
            atol=MESTI2S_3D_ATOL,
        )
        np.testing.assert_allclose(
            Ez,
            np.asarray(fixture["wavefront_field_Ez"], dtype=np.complex128),
            rtol=MESTI2S_3D_RTOL,
            atol=MESTI2S_3D_ATOL,
        )

    def test_3d_one_sided_nz_extension_wavefront_matches_julia_manual_fixture(self):
        fixture = _load_fixture(MESTI2S_3D_NZ_FIXTURE)
        syst = _three_d_system_from_fixture(fixture, prefix="one_", two_sided=False)

        Ex, Ey, Ez, channels, info = mesti2s(
            syst,
            wavefront(
                v_low_s=np.asarray(fixture["one_v_low_s"], dtype=np.complex128),
                v_low_p=np.asarray(fixture["one_v_low_p"], dtype=np.complex128),
            ),
            Opts(
                solver="scipy",
                verbal=False,
                nz_low=int(_scalar(fixture, "one_nz_low")),
                nz_high=int(_scalar(fixture, "one_nz_high")),
            ),
        )

        self.assertIsInstance(channels, Channels_one_sided)
        self.assertTrue(info.opts.return_field_profile)
        np.testing.assert_allclose(
            Ex,
            np.asarray(fixture["one_field_Ex"], dtype=np.complex128),
            rtol=MESTI2S_3D_RTOL,
            atol=MESTI2S_3D_ATOL,
        )
        np.testing.assert_allclose(
            Ey,
            np.asarray(fixture["one_field_Ey"], dtype=np.complex128),
            rtol=MESTI2S_3D_RTOL,
            atol=MESTI2S_3D_ATOL,
        )
        np.testing.assert_allclose(
            Ez,
            np.asarray(fixture["one_field_Ez"], dtype=np.complex128),
            rtol=MESTI2S_3D_RTOL,
            atol=MESTI2S_3D_ATOL,
        )
        np.testing.assert_allclose(Ex[:, :, -int(_scalar(fixture, "one_nz_high")) :, :], 0, atol=1e-12)

    def test_3d_mesti2s_unsupported_paths_are_explicit(self):
        fixture = _load_fixture(MESTI2S_3D_FIXTURE)
        syst = _three_d_system_from_fixture(fixture)

        offdiag_fixture = _load_fixture(MESTI2S_3D_OFFDIAGONAL_V5_FIXTURE)
        bad_shape = _three_d_system_from_fixture(offdiag_fixture, prefix="hermitian_")
        bad_shape.epsilon_xy = bad_shape.epsilon_xy[:, :, :-1]
        with self.assertRaisesRegex(ValueError, "epsilon_xy"):
            mesti2s(
                bad_shape,
                channel_type(side="both", polarization="both"),
                channel_type(side="both", polarization="both"),
                Opts(solver="scipy", verbal=False),
            )

        with self.assertRaises(NotImplementedError):
            mesti2s(
                syst,
                channel_type(side="low", polarization="s"),
                channel_type(side="low", polarization="s"),
                Opts(solver="scipy", verbal=False, symmetrize_K=True),
            )


if __name__ == "__main__":
    unittest.main()
