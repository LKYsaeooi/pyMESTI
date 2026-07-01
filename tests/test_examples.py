import unittest
from pathlib import Path
import importlib.util
import sys

import numpy as np
from scipy.io import loadmat

from mesti import (
    asp,
    build_epsilon_disorder,
    build_epsilon_disorder_3d,
    gaussian_beam_source_profiles,
    plot_and_compare_distribution,
    reflection_matrix_gaussian_beams,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
EXAMPLE_DIR = Path(__file__).parents[1] / "examples"
GAUSSIAN_REFLECTION_FIXTURE = FIXTURE_DIR / "example_reflection_gaussian_beams_v5.mat"
OPEN_CHANNEL_FIXTURE = FIXTURE_DIR / "example_open_channel_through_disorder_v5.mat"
PHASE_CONJUGATION_FIXTURE = FIXTURE_DIR / "example_focusing_phase_conjugation_v5.mat"
METALENS_ASP_FIXTURE = FIXTURE_DIR / "example_metalens_asp_v5.mat"
OPEN_CHANNEL_3D_FIXTURE = FIXTURE_DIR / "example_3d_open_channel_through_disorder_v5.mat"
EXAMPLE_RTOL = 5e-8
EXAMPLE_ATOL = 5e-9


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


def _vector(data, key, dtype=None):
    return np.asarray(data[key], dtype=dtype).reshape(-1)


def _string(data, key):
    return str(_scalar(data, key))


def _load_example_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module {filename}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _assert_allclose_up_to_global_phase(testcase, actual, expected, *, rtol, atol):
    actual_arr = np.asarray(actual, dtype=np.complex128)
    expected_arr = np.asarray(expected, dtype=np.complex128)
    phase_inner = np.vdot(actual_arr.reshape(-1), expected_arr.reshape(-1))
    if abs(phase_inner) > 0:
        actual_arr = actual_arr * (phase_inner / abs(phase_inner))
    testcase.assertEqual(actual_arr.shape, expected_arr.shape)
    np.testing.assert_allclose(actual_arr, expected_arr, rtol=rtol, atol=atol)


class MestiExamplesTest(unittest.TestCase):
    def test_asp_helper_matches_julia_fixture_slice(self):
        fixture = _load_fixture(METALENS_ASP_FIXTURE)

        actual = asp(
            np.asarray(fixture["field_right_after_metalens"], dtype=np.complex128)[:, 0],
            float(_scalar(fixture, "focal_length")),
            _vector(fixture, "kx_ASP_prop", dtype=np.complex128),
            int(_scalar(fixture, "ny_ASP")),
        )

        np.testing.assert_allclose(
            actual,
            np.asarray(fixture["field_at_focal_plane"], dtype=np.complex128)[:, 0],
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )

    def test_plot_and_compare_distribution_returns_julia_histogram_data(self):
        comparison = plot_and_compare_distribution([0.0, 0.01, 0.02, 0.51, 0.999, 1.0])

        self.assertEqual(comparison.bin_edges.size, 51)
        self.assertEqual(comparison.counts[0], 2)
        self.assertEqual(comparison.counts[1], 1)
        self.assertEqual(comparison.counts[-1], 2)
        self.assertAlmostEqual(float(np.sum(comparison.pdf) * comparison.bin_width), 1.0)
        np.testing.assert_allclose(
            comparison.dmpk_pdf,
            comparison.mean_tau / (2 * comparison.bin_centers * np.sqrt(1 - comparison.bin_centers)),
            rtol=0,
            atol=0,
        )

    def test_random_ball_disorder_builders_are_explicit_unsupported_stubs(self):
        with self.assertRaisesRegex(NotImplementedError, "Ball subpixel smoothing"):
            build_epsilon_disorder(
                4.0,
                5.0,
                0.1,
                0.2,
                0.05,
                0.1,
                1,
                0.1,
                1.44,
                1.0,
                True,
            )
        with self.assertRaisesRegex(NotImplementedError, "Ball subpixel smoothing"):
            build_epsilon_disorder_3d(
                3.0,
                4.0,
                5.0,
                0.1,
                0.2,
                0.05,
                0.1,
                1,
                0.1,
                1.44,
                1.0,
            )

    def test_gaussian_beam_source_profiles_match_julia_example_fixture(self):
        fixture = _load_fixture(GAUSSIAN_REFLECTION_FIXTURE)

        profiles = gaussian_beam_source_profiles(
            ny=int(_scalar(fixture, "ny_Ex")),
            y_coordinates=_vector(fixture, "y", dtype=float),
            y_focus=_vector(fixture, "y_f", dtype=float),
            z_source=float(_scalar(fixture, "z_s")),
            z_focus=float(_scalar(fixture, "z_f")),
            wavelength=float(_scalar(fixture, "wavelength")),
            dx=float(_scalar(fixture, "dx")),
            epsilon_bg=_scalar(fixture, "epsilon_bg"),
            numerical_aperture=float(_scalar(fixture, "NA")),
            yBC="PEC",
        )

        np.testing.assert_allclose(
            profiles.B_low,
            np.asarray(fixture["B_low"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            profiles.C_low,
            np.asarray(fixture["C_low"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(profiles.channels.N_prop, int(_scalar(fixture, "N_prop")))
        np.testing.assert_allclose(
            profiles.channels.kzdx_prop,
            _vector(fixture, "kzdx_prop", dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertLessEqual(
            profiles.transpose_mismatch_max,
            float(_scalar(fixture, "C_transpose_B_max_abs_difference")) + 1e-14,
        )

    def test_reflection_matrix_gaussian_beams_matches_julia_example_fixture(self):
        fixture = _load_fixture(GAUSSIAN_REFLECTION_FIXTURE)

        result = reflection_matrix_gaussian_beams(
            epsilon_xx=np.asarray(fixture["epsilon_xx"], dtype=np.complex128),
            wavelength=float(_scalar(fixture, "wavelength")),
            dx=float(_scalar(fixture, "dx")),
            pml_npixels=int(_scalar(fixture, "pml_npixels")),
            y_focus=_vector(fixture, "y_f", dtype=float),
            z_focus=float(_scalar(fixture, "z_f")),
            source_plane_index=int(_scalar(fixture, "source_plane_index_zero_based")),
            epsilon_bg=_scalar(fixture, "epsilon_bg"),
            numerical_aperture=float(_scalar(fixture, "NA")),
            y_coordinates=_vector(fixture, "y", dtype=float),
            z_coordinates=_vector(fixture, "z", dtype=float),
            solver="scipy",
        )

        self.assertFalse(result.reference_info.opts.return_field_profile)
        self.assertFalse(result.reflection_info.opts.return_field_profile)
        self.assertTrue(result.field_info.opts.return_field_profile)
        self.assertEqual(
            np.asarray(result.source.pos[0], dtype=int).tolist(),
            [0, int(_scalar(fixture, "source_plane_index_zero_based")), int(_scalar(fixture, "ny_Ex")) - 1, int(_scalar(fixture, "source_plane_index_zero_based"))],
        )
        np.testing.assert_allclose(
            result.reference,
            np.asarray(fixture["reference_D"], dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            result.reflection,
            np.asarray(fixture["reflection"], dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            np.abs(result.reflection) ** 2,
            np.asarray(fixture["reflection_abs_squared"], dtype=float),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            np.linalg.svd(result.reflection, compute_uv=False),
            _vector(fixture, "reflection_singular_values", dtype=float),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            result.field_profiles,
            np.asarray(fixture["field_profiles"], dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        self.assertEqual(
            list(result.field_profiles.shape),
            _vector(fixture, "field_profile_shape", dtype=int).tolist(),
        )

    def test_open_channel_through_disorder_example_matches_julia_fixture(self):
        fixture = _load_fixture(OPEN_CHANNEL_FIXTURE)
        example = _load_example_module(
            "open_channel_through_disorder.py",
            "open_channel_through_disorder_example",
        )

        result = example.run_open_channel_through_disorder(
            epsilon_xx=np.asarray(fixture["epsilon_xx"], dtype=np.complex128),
            wavelength=float(_scalar(fixture, "wavelength")),
            dx=float(_scalar(fixture, "dx")),
            epsilon_low=_scalar(fixture, "epsilon_low"),
            epsilon_high=_scalar(fixture, "epsilon_high"),
            yBC=_string(fixture, "yBC"),
            pml_npixels=int(_scalar(fixture, "pml_npixels")),
            nz_low=int(_scalar(fixture, "nz_low")),
            nz_high=int(_scalar(fixture, "nz_high")),
            solver="scipy",
            compute_direct_mesti_source=True,
        )

        self.assertFalse(result.transmission_info.opts.return_field_profile)
        self.assertTrue(result.field_info.opts.return_field_profile)
        self.assertTrue(result.direct_info.opts.return_field_profile)
        self.assertEqual(result.channels.low.N_prop, int(_scalar(fixture, "N_prop_low")))
        self.assertEqual(result.channels.high.N_prop, int(_scalar(fixture, "N_prop_high")))
        self.assertEqual(result.normal_index, int(_scalar(fixture, "normal_index_zero_based")))
        np.testing.assert_allclose(
            result.transmission,
            np.asarray(fixture["transmission"], dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            result.singular_values,
            _vector(fixture, "singular_values", dtype=float),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            result.transmission_eigenvalues,
            _vector(fixture, "transmission_eigenvalues", dtype=float),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        _assert_allclose_up_to_global_phase(
            self,
            result.open_channel,
            _vector(fixture, "open_channel", dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        self.assertAlmostEqual(result.average_transmission, float(_scalar(fixture, "T_avg")), places=10)
        self.assertAlmostEqual(result.plane_wave_transmission, float(_scalar(fixture, "T_PW")), places=10)
        self.assertAlmostEqual(result.open_channel_transmission, float(_scalar(fixture, "T_open")), places=10)
        np.testing.assert_allclose(
            result.field_profiles[:, :, 0],
            np.asarray(fixture["field_profiles"], dtype=np.complex128)[:, :, 0],
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        _assert_allclose_up_to_global_phase(
            self,
            result.field_profiles[:, :, 1],
            np.asarray(fixture["field_profiles"], dtype=np.complex128)[:, :, 1],
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        self.assertEqual(
            list(result.field_profiles.shape),
            _vector(fixture, "field_profile_shape", dtype=int).tolist(),
        )
        self.assertLessEqual(result.direct_field_difference_max, 5e-8)
        self.assertLessEqual(
            result.direct_field_difference_max,
            float(_scalar(fixture, "direct_field_difference_max")) + 5e-8,
        )

    def test_focusing_inside_disorder_phase_conjugation_example_matches_julia_fixture(self):
        fixture = _load_fixture(PHASE_CONJUGATION_FIXTURE)
        example = _load_example_module(
            "focusing_inside_disorder_with_phase_conjugation.py",
            "focusing_inside_disorder_with_phase_conjugation_example",
        )

        result = example.run_focusing_inside_disorder_with_phase_conjugation(
            epsilon_xx=np.asarray(fixture["epsilon_xx"], dtype=np.complex128),
            wavelength=float(_scalar(fixture, "wavelength")),
            dx=float(_scalar(fixture, "dx")),
            epsilon_low=_scalar(fixture, "epsilon_low"),
            epsilon_high=_scalar(fixture, "epsilon_high"),
            yBC=_string(fixture, "yBC"),
            pml_npixels=int(_scalar(fixture, "pml_npixels")),
            nz_low=int(_scalar(fixture, "nz_low")),
            nz_high=int(_scalar(fixture, "nz_high")),
            focus_index=(
                int(_scalar(fixture, "focus_y_index_zero_based")),
                int(_scalar(fixture, "focus_z_index_zero_based")),
            ),
            solver="scipy",
            compute_field_projection_check=True,
        )

        self.assertFalse(result.projection_info.opts.return_field_profile)
        self.assertTrue(result.projection_field_info.opts.return_field_profile)
        self.assertTrue(result.field_info.opts.return_field_profile)
        self.assertEqual(result.channels_low.N_prop, int(_scalar(fixture, "N_prop_low")))
        self.assertEqual(result.channels_average.N_prop, int(_scalar(fixture, "N_prop_ave_epsilon")))
        self.assertEqual(
            result.focus_index,
            (
                int(_scalar(fixture, "focus_y_index_zero_based")),
                int(_scalar(fixture, "focus_z_index_zero_based")),
            ),
        )
        np.testing.assert_allclose(
            result.projected_coefficients,
            _vector(fixture, "projected_coefficients", dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            result.projection_from_field,
            _vector(fixture, "projection_from_field", dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        self.assertLessEqual(result.projection_from_field_difference_max, 5e-8)
        self.assertLessEqual(
            result.projection_from_field_difference_max,
            float(_scalar(fixture, "projection_from_field_difference_max")) + 5e-8,
        )
        np.testing.assert_allclose(
            result.regular_focus_wavefront,
            _vector(fixture, "regular_focus_wavefront", dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            result.phase_conjugated_wavefront,
            _vector(fixture, "phase_conjugated_wavefront", dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            result.v_low,
            np.asarray(fixture["v_low"], dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            result.field_profiles,
            np.asarray(fixture["field_profiles"], dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            result.normalized_field_profiles,
            np.asarray(fixture["normalized_field_profiles"], dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        self.assertEqual(
            list(result.field_profiles.shape),
            _vector(fixture, "field_profile_shape", dtype=int).tolist(),
        )
        self.assertAlmostEqual(result.normalization_factor, float(_scalar(fixture, "normalization_factor")), places=10)
        np.testing.assert_allclose(
            result.focus_intensities,
            _vector(fixture, "focus_intensities", dtype=float),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        self.assertAlmostEqual(
            result.phase_to_regular_intensity_ratio,
            float(_scalar(fixture, "phase_to_regular_intensity_ratio")),
            places=10,
        )

    def test_metalens_angular_spectrum_example_matches_julia_fixture(self):
        fixture = _load_fixture(METALENS_ASP_FIXTURE)
        example = _load_example_module(
            "metalens_focusing_via_angular_spectrum_propagation.py",
            "metalens_focusing_via_angular_spectrum_propagation_example",
        )

        result = example.run_metalens_focusing_via_angular_spectrum_propagation(
            epsilon_metalens=np.asarray(fixture["epsilon_metalens"], dtype=np.complex128),
            n_air=float(_scalar(fixture, "n_air")),
            n_sub=float(_scalar(fixture, "n_sub")),
            n_struct=float(_scalar(fixture, "n_struct")),
            wavelength=float(_scalar(fixture, "wavelength")),
            dx=float(_scalar(fixture, "dx")),
            d_out=float(_scalar(fixture, "D_out")),
            d_in=float(_scalar(fixture, "D_in")),
            h=float(_scalar(fixture, "h")),
            numerical_aperture=float(_scalar(fixture, "NA")),
            w_out=float(_scalar(fixture, "W_out")),
            theta_in_list=_vector(fixture, "theta_in_list", dtype=float),
            pml_npixels=int(_scalar(fixture, "nPML")),
            dy_asp=float(_scalar(fixture, "dy_ASP_input")),
            ybc_channels=_string(fixture, "yBC_channels"),
            use_continuous_dispersion=bool(_scalar(fixture, "use_continuous_dispersion")),
            solver="scipy",
        )

        self.assertFalse(result.direct_info.opts.return_field_profile)
        self.assertEqual(result.channels_left.N_prop, int(_scalar(fixture, "N_prop_L")))
        self.assertEqual(
            np.asarray(result.source.pos[0], dtype=int).tolist(),
            _vector(fixture, "source_pos_zero_based_inclusive", dtype=int).tolist(),
        )
        self.assertEqual(
            np.asarray(result.projection.pos[0], dtype=int).tolist(),
            _vector(fixture, "projection_pos_zero_based_inclusive", dtype=int).tolist(),
        )
        np.testing.assert_allclose(
            result.epsilon_syst,
            np.asarray(fixture["epsilon_syst"], dtype=np.complex128),
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            result.b_trunc,
            np.asarray(fixture["B_trunc"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.b_left,
            np.asarray(fixture["B_L"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.c_right,
            np.asarray(fixture["C_R"], dtype=np.complex128),
            rtol=0,
            atol=0,
        )
        self.assertEqual(result.asp_setup.ny_asp, int(_scalar(fixture, "ny_ASP")))
        self.assertEqual(result.asp_setup.ind_asp.tolist(), _vector(fixture, "ind_ASP_zero_based", dtype=int).tolist())
        self.assertEqual(
            result.asp_setup.prop_indices.tolist(),
            _vector(fixture, "asp_prop_indices_zero_based", dtype=int).tolist(),
        )
        np.testing.assert_allclose(
            result.asp_setup.y_asp,
            _vector(fixture, "y_ASP", dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.asp_setup.kx_asp_prop,
            _vector(fixture, "kx_ASP_prop", dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.field_right_after_metalens,
            np.asarray(fixture["field_right_after_metalens"], dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            result.field_at_focal_plane,
            np.asarray(fixture["field_at_focal_plane"], dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            result.focal_plane_intensity,
            np.asarray(fixture["focal_plane_intensity"], dtype=float),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        self.assertEqual(
            result.target_focal_indices.tolist(),
            _vector(fixture, "target_focal_indices_zero_based", dtype=int).tolist(),
        )
        self.assertEqual(
            result.peak_indices.tolist(),
            _vector(fixture, "peak_indices_zero_based", dtype=int).tolist(),
        )
        np.testing.assert_allclose(
            result.target_focal_intensities,
            _vector(fixture, "target_focal_intensities", dtype=float),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            result.peak_intensities,
            _vector(fixture, "peak_intensities", dtype=float),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            result.peak_y_positions,
            _vector(fixture, "peak_y_positions", dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_open_channel_through_disorder_3d_example_matches_julia_fixture(self):
        fixture = _load_fixture(OPEN_CHANNEL_3D_FIXTURE)
        example = _load_example_module(
            "open_channel_through_disorder_3d.py",
            "open_channel_through_disorder_3d_example",
        )

        result = example.run_open_channel_through_disorder_3d(
            epsilon_xx=np.asarray(fixture["epsilon_xx"], dtype=np.complex128),
            epsilon_yy=np.asarray(fixture["epsilon_yy"], dtype=np.complex128),
            epsilon_zz=np.asarray(fixture["epsilon_zz"], dtype=np.complex128),
            wavelength=float(_scalar(fixture, "wavelength")),
            dx=float(_scalar(fixture, "dx")),
            epsilon_low=_scalar(fixture, "epsilon_low"),
            epsilon_high=_scalar(fixture, "epsilon_high"),
            xBC=_string(fixture, "xBC"),
            yBC=_string(fixture, "yBC"),
            pml_npixels=int(_scalar(fixture, "zPML_npixels")),
            solver="scipy",
        )

        self.assertFalse(result.transmission_info.opts.return_field_profile)
        self.assertTrue(result.field_info.opts.return_field_profile)
        self.assertEqual(result.channels.low.N_prop, int(_scalar(fixture, "low_N_prop")))
        self.assertEqual(result.channels.high.N_prop, int(_scalar(fixture, "high_N_prop")))
        self.assertEqual(result.normal_index, int(_scalar(fixture, "normal_index_zero_based")))
        self.assertEqual(result.normal_s_column, int(_scalar(fixture, "normal_s_column_zero_based")))
        self.assertEqual(result.normal_p_column, int(_scalar(fixture, "normal_p_column_zero_based")))
        np.testing.assert_allclose(
            result.transmission,
            np.asarray(fixture["transmission"], dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            result.singular_values,
            _vector(fixture, "singular_values", dtype=float),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        np.testing.assert_allclose(
            result.transmission_eigenvalues,
            _vector(fixture, "transmission_eigenvalues", dtype=float),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        _assert_allclose_up_to_global_phase(
            self,
            result.open_channel,
            _vector(fixture, "open_channel", dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        _assert_allclose_up_to_global_phase(
            self,
            result.closed_channel,
            _vector(fixture, "closed_channel", dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        self.assertAlmostEqual(result.average_transmission, float(_scalar(fixture, "T_avg")), places=10)
        self.assertAlmostEqual(result.plane_wave_s_transmission, float(_scalar(fixture, "T_PW_s")), places=10)
        self.assertAlmostEqual(result.plane_wave_p_transmission, float(_scalar(fixture, "T_PW_p")), places=10)
        self.assertAlmostEqual(result.closed_channel_transmission, float(_scalar(fixture, "T_closed")), places=10)
        self.assertAlmostEqual(result.open_channel_transmission, float(_scalar(fixture, "T_open")), places=10)

        expected_v_low_p = np.asarray(fixture["v_low_p"], dtype=np.complex128)
        np.testing.assert_allclose(result.v_low_p[:, 2], expected_v_low_p[:, 2], rtol=0, atol=0)
        _assert_allclose_up_to_global_phase(
            self,
            np.r_[result.v_low_s[:, 0], result.v_low_p[:, 0]],
            _vector(fixture, "closed_channel", dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )
        _assert_allclose_up_to_global_phase(
            self,
            np.r_[result.v_low_s[:, 1], result.v_low_p[:, 1]],
            _vector(fixture, "open_channel", dtype=np.complex128),
            rtol=EXAMPLE_RTOL,
            atol=EXAMPLE_ATOL,
        )

        self.assertEqual(
            list(result.field_Ex.shape),
            _vector(fixture, "field_profile_shape_Ex", dtype=int).tolist(),
        )
        self.assertEqual(
            list(result.field_Ey.shape),
            _vector(fixture, "field_profile_shape_Ey", dtype=int).tolist(),
        )
        self.assertEqual(
            list(result.field_Ez.shape),
            _vector(fixture, "field_profile_shape_Ez", dtype=int).tolist(),
        )
        expected_ex = np.asarray(fixture["field_Ex"], dtype=np.complex128)
        expected_ey = np.asarray(fixture["field_Ey"], dtype=np.complex128)
        expected_ez = np.asarray(fixture["field_Ez"], dtype=np.complex128)
        np.testing.assert_allclose(result.field_Ex[:, :, :, 4], expected_ex[:, :, :, 4], rtol=EXAMPLE_RTOL, atol=EXAMPLE_ATOL)
        np.testing.assert_allclose(result.field_Ey[:, :, :, 4], expected_ey[:, :, :, 4], rtol=EXAMPLE_RTOL, atol=EXAMPLE_ATOL)
        np.testing.assert_allclose(result.field_Ez[:, :, :, 4], expected_ez[:, :, :, 4], rtol=EXAMPLE_RTOL, atol=EXAMPLE_ATOL)
        for actual, expected_key in (
            (result.combined_closed_Ex, "combined_closed_Ex"),
            (result.combined_closed_Ey, "combined_closed_Ey"),
            (result.combined_closed_Ez, "combined_closed_Ez"),
            (result.combined_open_Ex, "combined_open_Ex"),
            (result.combined_open_Ey, "combined_open_Ey"),
            (result.combined_open_Ez, "combined_open_Ez"),
        ):
            _assert_allclose_up_to_global_phase(
                self,
                actual,
                np.asarray(fixture[expected_key], dtype=np.complex128),
                rtol=EXAMPLE_RTOL,
                atol=EXAMPLE_ATOL,
            )
        self.assertAlmostEqual(
            result.open_ex_normalization_factor,
            float(_scalar(fixture, "open_ex_normalization_factor")),
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
