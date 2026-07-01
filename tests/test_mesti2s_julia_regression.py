import unittest
import importlib.util
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from mesti import Opts, PML, Syst, channel_index, channel_type, mesti, mesti2s, wavefront


FIXTURE_DIR = Path(__file__).parent / "fixtures"
LOW_TO_HIGH_FIXTURE = FIXTURE_DIR / "mesti2s_2d_tm_julia_low_to_high.mat"
WAVEFRONT_FIXTURE = FIXTURE_DIR / "mesti2s_2d_tm_julia_wavefront_v_low.mat"
WS30_CROPPED_FIXTURE = FIXTURE_DIR / "mesti2s_2d_tm_ws30_center384_double_mumps.mat"
STEP4_BLOCH_FIXTURE = FIXTURE_DIR / "mesti2s_2d_tm_step4_bloch_continuous.mat"
STEP4_NONPERIODIC_FIXTURE = FIXTURE_DIR / "mesti2s_2d_tm_step4_nonperiodic.mat"
STEP4_SPACER_FIXTURE = FIXTURE_DIR / "mesti2s_2d_tm_step4_spacer_wavefront.mat"
STEP4_INTERFACE_FIXTURE = FIXTURE_DIR / "mesti2s_2d_tm_step4_interface_rt.mat"
STEP4_DIRECT_MESTI_FIXTURE = FIXTURE_DIR / "mesti_step4_direct_2d_tm.mat"
JULIA_PARITY_RTOL = 5e-5
JULIA_PARITY_ATOL = 2e-6
DOUBLE_MUMPS_PARITY_RTOL = 5e-10
DOUBLE_MUMPS_PARITY_ATOL = 5e-11
STEP4_PARITY_RTOL = 5e-8
STEP4_PARITY_ATOL = 5e-9


def _load_fixture(path):
    try:
        return {
            key: value
            for key, value in loadmat(path, squeeze_me=False).items()
            if not key.startswith("__")
        }
    except NotImplementedError:
        return _load_hdf5_fixture(path)


def _load_hdf5_fixture(path):
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
    value = _scalar(data, key)
    return str(value)


def _vector(data, key, dtype=None):
    return np.asarray(data[key], dtype=dtype).reshape(-1)


def _bool(data, key):
    return bool(_scalar(data, key))


def _pml_layers_from_fixture(data):
    low = PML(
        int(_scalar(data, "zPML_npixels")),
        npixels_spacer=int(_scalar(data, "zPML_npixels_spacer")) if "zPML_npixels_spacer" in data else None,
    )
    if "zPML_high_npixels" not in data:
        return [low]
    high = PML(
        int(_scalar(data, "zPML_high_npixels")),
        npixels_spacer=(
            int(_scalar(data, "zPML_high_npixels_spacer"))
            if "zPML_high_npixels_spacer" in data
            else None
        ),
    )
    return [low, high]


def _system_from_fixture(data):
    syst = Syst(
        epsilon_xx=np.asarray(data["epsilon_xx"], dtype=np.complex128),
        epsilon_low=float(_scalar(data, "epsilon_low")) if "epsilon_low" in data else None,
        epsilon_high=float(_scalar(data, "epsilon_high")) if "epsilon_high" in data else None,
        wavelength=float(_scalar(data, "wavelength")),
        dx=float(_scalar(data, "dx")),
        yBC=_string(data, "yBC"),
        zPML=_pml_layers_from_fixture(data) if "zPML_npixels" in data else None,
    )
    if "ky_B" in data:
        syst.ky_B = float(_scalar(data, "ky_B"))
    if "zBC" in data:
        syst.zBC = _string(data, "zBC")
    return syst


def _assert_two_sided_metadata(testcase, channels, fixture):
    testcase.assertEqual(channels.low.N_prop, int(_scalar(fixture, "low_N_prop")))
    testcase.assertEqual(channels.high.N_prop, int(_scalar(fixture, "high_N_prop")))
    testcase.assertEqual(
        np.asarray(channels.low.ind_prop).tolist(),
        _vector(fixture, "low_ind_prop_zero_based", dtype=int).tolist(),
    )
    testcase.assertEqual(
        np.asarray(channels.high.ind_prop).tolist(),
        _vector(fixture, "high_ind_prop_zero_based", dtype=int).tolist(),
    )
    np.testing.assert_allclose(
        channels.kydx_all,
        _vector(fixture, "kydx_all", dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        channels.low.kzdx_prop,
        _vector(fixture, "low_kzdx_prop", dtype=np.complex128),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        channels.high.kzdx_prop,
        _vector(fixture, "high_kzdx_prop", dtype=np.complex128),
        rtol=1e-10,
        atol=1e-10,
    )


def _available_mumps_solver():
    if importlib.util.find_spec("mumpspy") is not None:
        return "mumpspy"
    if importlib.util.find_spec("mumps") is not None:
        return "python-mumps"
    return None


class Mesti2SJuliaRegressionFixtureTest(unittest.TestCase):
    def test_julia_fixture_low_to_high_transmission(self):
        fixture = _load_fixture(LOW_TO_HIGH_FIXTURE)
        syst = _system_from_fixture(fixture)

        t, channels, info = mesti2s(
            syst,
            channel_type(side="low"),
            channel_type(side="high"),
            Opts(verbal=False),
        )

        self.assertEqual(int(_scalar(fixture, "fixture_format")), 1)
        self.assertEqual(channels.low.N_prop, int(_scalar(fixture, "low_N_prop")))
        self.assertEqual(channels.high.N_prop, int(_scalar(fixture, "high_N_prop")))
        self.assertEqual(
            np.asarray(channels.low.ind_prop).tolist(),
            _vector(fixture, "low_ind_prop_zero_based", dtype=int).tolist(),
        )
        self.assertEqual(
            np.asarray(channels.high.ind_prop).tolist(),
            _vector(fixture, "high_ind_prop_zero_based", dtype=int).tolist(),
        )
        self.assertFalse(info.opts.return_field_profile)
        self.assertEqual(bool(_scalar(fixture, "return_field_profile")), info.opts.return_field_profile)
        np.testing.assert_allclose(
            channels.low.kzdx_prop,
            _vector(fixture, "low_kzdx_prop", dtype=np.complex128),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            channels.high.kzdx_prop,
            _vector(fixture, "high_kzdx_prop", dtype=np.complex128),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            t,
            np.asarray(fixture["t"], dtype=np.complex128),
            rtol=JULIA_PARITY_RTOL,
            atol=JULIA_PARITY_ATOL,
        )
        np.testing.assert_allclose(
            np.linalg.svd(t, compute_uv=False),
            _vector(fixture, "singular_values", dtype=float),
            rtol=JULIA_PARITY_RTOL,
            atol=JULIA_PARITY_ATOL,
        )

    def test_julia_fixture_channel_index_low_to_high_submatrix(self):
        fixture = _load_fixture(LOW_TO_HIGH_FIXTURE)
        syst = _system_from_fixture(fixture)
        selected_low = np.array([0, 2], dtype=int)
        selected_high = np.array([1, 2], dtype=int)

        t_subset, channels, info = mesti2s(
            syst,
            channel_index(ind_low=selected_low),
            channel_index(ind_high=selected_high),
            Opts(verbal=False),
        )

        expected = np.asarray(fixture["t"], dtype=np.complex128)[np.ix_(selected_high, selected_low)]
        self.assertEqual(channels.low.N_prop, int(_scalar(fixture, "low_N_prop")))
        self.assertEqual(channels.high.N_prop, int(_scalar(fixture, "high_N_prop")))
        self.assertEqual(t_subset.shape, expected.shape)
        self.assertFalse(info.opts.return_field_profile)
        np.testing.assert_allclose(
            t_subset,
            expected,
            rtol=JULIA_PARITY_RTOL,
            atol=JULIA_PARITY_ATOL,
        )

    def test_julia_fixture_wavefront_v_low_to_high_projection(self):
        fixture = _load_fixture(LOW_TO_HIGH_FIXTURE)
        wave_fixture = _load_fixture(WAVEFRONT_FIXTURE)
        syst = _system_from_fixture(fixture)
        t = np.asarray(fixture["t"], dtype=np.complex128)
        v_low = np.asarray(wave_fixture["v_low"], dtype=np.complex128)

        projected, channels, info = mesti2s(
            syst,
            wavefront(v_low=v_low),
            channel_type(side="high"),
            Opts(verbal=False),
        )

        expected = t @ v_low
        self.assertEqual(channels.low.N_prop, int(_scalar(fixture, "low_N_prop")))
        self.assertEqual(channels.high.N_prop, int(_scalar(fixture, "high_N_prop")))
        self.assertEqual(projected.shape, expected.shape)
        self.assertFalse(info.opts.return_field_profile)
        np.testing.assert_allclose(
            projected,
            expected,
            rtol=JULIA_PARITY_RTOL,
            atol=JULIA_PARITY_ATOL,
        )

    def test_julia_fixture_wavefront_v_high_output_projection(self):
        fixture = _load_fixture(LOW_TO_HIGH_FIXTURE)
        syst = _system_from_fixture(fixture)
        t = np.asarray(fixture["t"], dtype=np.complex128)
        v_high = np.array(
            [
                [1.0 + 0.0j, 0.2 - 0.1j],
                [0.3 + 0.4j, -0.5j],
                [-0.2j, 0.25 + 0.0j],
            ],
            dtype=np.complex128,
        )

        projected, channels, info = mesti2s(
            syst,
            channel_type(side="low"),
            wavefront(v_high=v_high),
            Opts(verbal=False),
        )

        expected = v_high.conjugate().T @ t
        self.assertEqual(channels.low.N_prop, int(_scalar(fixture, "low_N_prop")))
        self.assertEqual(channels.high.N_prop, int(_scalar(fixture, "high_N_prop")))
        self.assertEqual(projected.shape, expected.shape)
        self.assertFalse(info.opts.return_field_profile)
        np.testing.assert_allclose(
            projected,
            expected,
            rtol=JULIA_PARITY_RTOL,
            atol=JULIA_PARITY_ATOL,
        )

    def test_julia_fixture_wavefront_v_low_field_profile(self):
        fixture = _load_fixture(WAVEFRONT_FIXTURE)
        syst = _system_from_fixture(fixture)
        v_low = np.asarray(fixture["v_low"], dtype=np.complex128)

        field_profile, channels, info = mesti2s(
            syst,
            wavefront(v_low=v_low),
            Opts(verbal=False),
        )

        self.assertEqual(int(_scalar(fixture, "fixture_format")), 1)
        self.assertEqual(channels.low.N_prop, int(_scalar(fixture, "low_N_prop")))
        self.assertTrue(info.opts.return_field_profile)
        self.assertEqual(bool(_scalar(fixture, "return_field_profile")), info.opts.return_field_profile)
        np.testing.assert_allclose(
            field_profile,
            np.asarray(fixture["field_profile"], dtype=np.complex128),
            rtol=JULIA_PARITY_RTOL,
            atol=JULIA_PARITY_ATOL,
        )

    def test_ws30_centered_crop_matches_julia_double_mumps_fixture(self):
        solver = _available_mumps_solver()
        if solver is None:
            self.skipTest("cropped-real Julia parity fixture requires a Python MUMPS binding")

        fixture = _load_fixture(WS30_CROPPED_FIXTURE)
        syst = _system_from_fixture(fixture)
        opts = Opts(solver=solver, nrhs=95, verbal=False)

        t, channels, info = mesti2s(
            syst,
            channel_type(side="low"),
            channel_type(side="high"),
            opts,
        )

        self.assertEqual(int(_scalar(fixture, "fixture_format")), 1)
        self.assertFalse(bool(_scalar(fixture, "use_single_precision_MUMPS")))
        self.assertEqual(channels.low.N_prop, int(_scalar(fixture, "low_N_prop")))
        self.assertEqual(channels.high.N_prop, int(_scalar(fixture, "high_N_prop")))
        self.assertFalse(info.opts.return_field_profile)
        np.testing.assert_allclose(
            t,
            np.asarray(fixture["t"], dtype=np.complex128),
            rtol=DOUBLE_MUMPS_PARITY_RTOL,
            atol=DOUBLE_MUMPS_PARITY_ATOL,
        )
        np.testing.assert_allclose(
            np.linalg.svd(t, compute_uv=False),
            _vector(fixture, "singular_values", dtype=float),
            rtol=DOUBLE_MUMPS_PARITY_RTOL,
            atol=DOUBLE_MUMPS_PARITY_ATOL,
        )

        field_profile, channels_field, info_field = mesti2s(
            syst,
            wavefront(v_low=np.asarray(fixture["v_low"], dtype=np.complex128)),
            Opts(solver=solver, nrhs=95, verbal=False),
        )

        self.assertEqual(channels_field.low.N_prop, channels.low.N_prop)
        self.assertTrue(info_field.opts.return_field_profile)
        np.testing.assert_allclose(
            field_profile,
            np.asarray(fixture["field_profile"], dtype=np.complex128),
            rtol=DOUBLE_MUMPS_PARITY_RTOL,
            atol=DOUBLE_MUMPS_PARITY_ATOL,
        )

    def test_step4_bloch_ky_b_continuous_m0_matches_julia_fixture(self):
        fixture = _load_fixture(STEP4_BLOCH_FIXTURE)
        syst = _system_from_fixture(fixture)
        opts = Opts(
            verbal=False,
            use_continuous_dispersion=_bool(fixture, "use_continuous_dispersion"),
            m0=float(_scalar(fixture, "m0")),
        )

        S, channels, info = mesti2s(
            syst,
            channel_type(side="low"),
            channel_type(side="high"),
            opts,
        )

        self.assertEqual(_string(fixture, "yBC"), "Bloch")
        self.assertIsNotNone(syst.ky_B)
        self.assertFalse(info.opts.return_field_profile)
        self.assertFalse(_bool(fixture, "return_field_profile"))
        _assert_two_sided_metadata(self, channels, fixture)
        np.testing.assert_allclose(
            S,
            np.asarray(fixture["S"], dtype=np.complex128),
            rtol=STEP4_PARITY_RTOL,
            atol=STEP4_PARITY_ATOL,
        )
        np.testing.assert_allclose(
            np.linalg.svd(S, compute_uv=False),
            _vector(fixture, "singular_values", dtype=float),
            rtol=STEP4_PARITY_RTOL,
            atol=STEP4_PARITY_ATOL,
        )

    def test_step4_nonperiodic_transverse_boundary_matches_julia_fixture(self):
        fixture = _load_fixture(STEP4_NONPERIODIC_FIXTURE)
        syst = _system_from_fixture(fixture)

        S, channels, info = mesti2s(
            syst,
            channel_type(side="low"),
            channel_type(side="high"),
            Opts(verbal=False),
        )

        self.assertEqual(_string(fixture, "yBC"), "PMC")
        self.assertFalse(info.opts.return_field_profile)
        _assert_two_sided_metadata(self, channels, fixture)
        np.testing.assert_allclose(
            S,
            np.asarray(fixture["S"], dtype=np.complex128),
            rtol=STEP4_PARITY_RTOL,
            atol=STEP4_PARITY_ATOL,
        )
        np.testing.assert_allclose(
            np.linalg.svd(S, compute_uv=False),
            _vector(fixture, "singular_values", dtype=float),
            rtol=STEP4_PARITY_RTOL,
            atol=STEP4_PARITY_ATOL,
        )

    def test_step4_spacer_both_sides_and_mixed_wavefront_match_julia_fixture(self):
        fixture = _load_fixture(STEP4_SPACER_FIXTURE)
        syst = _system_from_fixture(fixture)

        S_both, channels, info = mesti2s(
            syst,
            channel_type(side="both"),
            channel_type(side="both"),
            Opts(verbal=False),
        )

        self.assertEqual(int(_scalar(fixture, "zPML_npixels_spacer")), 1)
        self.assertEqual(int(_scalar(fixture, "zPML_high_npixels_spacer")), 2)
        self.assertFalse(info.opts.return_field_profile)
        self.assertFalse(_bool(fixture, "return_field_profile_S"))
        _assert_two_sided_metadata(self, channels, fixture)
        np.testing.assert_allclose(
            S_both,
            np.asarray(fixture["S_both"], dtype=np.complex128),
            rtol=STEP4_PARITY_RTOL,
            atol=STEP4_PARITY_ATOL,
        )
        np.testing.assert_allclose(
            np.linalg.svd(S_both, compute_uv=False),
            _vector(fixture, "S_both_singular_values", dtype=float),
            rtol=STEP4_PARITY_RTOL,
            atol=STEP4_PARITY_ATOL,
        )

        field_profile, channels_field, info_field = mesti2s(
            syst,
            wavefront(
                v_low=np.asarray(fixture["v_low"], dtype=np.complex128),
                v_high=np.asarray(fixture["v_high"], dtype=np.complex128),
            ),
            Opts(
                verbal=False,
                nz_low=int(_scalar(fixture, "nz_low")),
                nz_high=int(_scalar(fixture, "nz_high")),
            ),
        )

        self.assertEqual(channels_field.low.N_prop, channels.low.N_prop)
        self.assertEqual(channels_field.high.N_prop, channels.high.N_prop)
        self.assertTrue(info_field.opts.return_field_profile)
        self.assertTrue(_bool(fixture, "return_field_profile_field"))
        np.testing.assert_allclose(
            field_profile,
            np.asarray(fixture["field_profile"], dtype=np.complex128),
            rtol=STEP4_PARITY_RTOL,
            atol=STEP4_PARITY_ATOL,
        )

    def test_step4_interface_reflection_transmission_matches_julia_fixture(self):
        fixture = _load_fixture(STEP4_INTERFACE_FIXTURE)
        syst = Syst(
            epsilon_xx=np.ones((1, 0), dtype=np.complex128),
            epsilon_low=float(_scalar(fixture, "epsilon_low")),
            epsilon_high=float(_scalar(fixture, "epsilon_high")),
            wavelength=float(_scalar(fixture, "wavelength")),
            dx=float(_scalar(fixture, "dx")),
            yBC=_string(fixture, "yBC"),
            zPML=[PML(int(_scalar(fixture, "zPML_npixels")))],
        )

        S, channels, info = mesti2s(
            syst,
            channel_type(side="low"),
            channel_type(side="both"),
            Opts(verbal=False),
        )

        self.assertEqual(channels.low.N_prop, 1)
        self.assertEqual(channels.high.N_prop, 1)
        self.assertEqual(S.shape, (2, 1))
        self.assertFalse(info.opts.return_field_profile)
        np.testing.assert_allclose(
            S,
            np.asarray(fixture["S"], dtype=np.complex128),
            rtol=STEP4_PARITY_RTOL,
            atol=STEP4_PARITY_ATOL,
        )
        np.testing.assert_allclose(
            S[0, 0],
            _scalar(fixture, "r_analytic"),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            S[1, 0],
            _scalar(fixture, "t_analytic"),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_step4_direct_mesti_field_and_projection_match_julia_fixture(self):
        fixture = _load_fixture(STEP4_DIRECT_MESTI_FIXTURE)
        syst = _system_from_fixture(fixture)
        B = np.asarray(fixture["B"], dtype=np.complex128)
        C = np.asarray(fixture["C"], dtype=np.complex128)

        field_profile, info_field = mesti(syst, B, opts=Opts(verbal=False))
        projection, info_projection = mesti(syst, B, C=C, opts=Opts(verbal=False))

        self.assertTrue(info_field.opts.return_field_profile)
        self.assertFalse(info_projection.opts.return_field_profile)
        np.testing.assert_allclose(
            field_profile,
            np.asarray(fixture["field_profile"], dtype=np.complex128),
            rtol=STEP4_PARITY_RTOL,
            atol=STEP4_PARITY_ATOL,
        )
        np.testing.assert_allclose(
            projection,
            np.asarray(fixture["projection"], dtype=np.complex128),
            rtol=STEP4_PARITY_RTOL,
            atol=STEP4_PARITY_ATOL,
        )


if __name__ == "__main__":
    unittest.main()
