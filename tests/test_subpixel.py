import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from mesti import Ball, Cuboid, mesti_subpixel_smoothing


FIXTURE_DIR = Path(__file__).parent / "fixtures"
SUBPIXEL_2D_TM_V5_FIXTURE = FIXTURE_DIR / "subpixel_2d_tm_v5.mat"
SUBPIXEL_3D_CUBOID_V7_FIXTURE = FIXTURE_DIR / "subpixel_3d_cuboid_v7.mat"
SUBPIXEL_BALL_B1_FIXTURE = FIXTURE_DIR / "subpixel_ball_b1.mat"
SUBPIXEL_3D_COMPONENT_KEYS = (
    "epsilon_xx",
    "epsilon_xy",
    "epsilon_xz",
    "epsilon_yx",
    "epsilon_yy",
    "epsilon_yz",
    "epsilon_zx",
    "epsilon_zy",
    "epsilon_zz",
)


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


def _cuboid_from_fixture(data, prefix):
    return Cuboid(
        np.asarray(data[f"{prefix}_center"], dtype=float).reshape(-1),
        np.asarray(data[f"{prefix}_widths"], dtype=float).reshape(-1),
    )


def _ball_from_fixture(data, prefix):
    return Ball(
        np.asarray(data[f"{prefix}_center"], dtype=float).reshape(-1),
        float(_scalar(data, f"{prefix}_radius")),
    )


def _assert_3d_cuboid_fixture_matches(test_case, data, prefix, *, without_sb=False):
    components = mesti_subpixel_smoothing(
        float(_scalar(data, f"{prefix}_delta_x")),
        _cuboid_from_fixture(data, f"{prefix}_domain"),
        _scalar(data, f"{prefix}_domain_epsilon"),
        [_cuboid_from_fixture(data, f"{prefix}_object")],
        [_scalar(data, f"{prefix}_object_epsilon")],
        _string(data, f"{prefix}_xBC"),
        _string(data, f"{prefix}_yBC"),
        _string(data, f"{prefix}_zBC"),
        without_sb=without_sb,
    )

    suffix = "_without_sb" if without_sb else ""
    test_case.assertEqual(len(components), len(SUBPIXEL_3D_COMPONENT_KEYS))
    for actual, key in zip(components, SUBPIXEL_3D_COMPONENT_KEYS):
        np.testing.assert_allclose(
            actual,
            np.asarray(data[f"{prefix}_{key}{suffix}"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )


def _assert_3d_ball_fixture_matches(test_case, data, prefix, *, without_sb=False):
    components = mesti_subpixel_smoothing(
        float(_scalar(data, f"{prefix}_delta_x")),
        _cuboid_from_fixture(data, f"{prefix}_domain"),
        _scalar(data, f"{prefix}_domain_epsilon"),
        [_ball_from_fixture(data, f"{prefix}_object")],
        [_scalar(data, f"{prefix}_object_epsilon")],
        _string(data, f"{prefix}_xBC"),
        _string(data, f"{prefix}_yBC"),
        _string(data, f"{prefix}_zBC"),
        without_sb=without_sb,
    )

    suffix = "_without_sb" if without_sb else ""
    test_case.assertEqual(len(components), len(SUBPIXEL_3D_COMPONENT_KEYS))
    for actual, key in zip(components, SUBPIXEL_3D_COMPONENT_KEYS):
        np.testing.assert_allclose(
            actual,
            np.asarray(data[f"{prefix}_{key}{suffix}"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )


class SubpixelSmoothingTest(unittest.TestCase):
    def test_ball_compatibility_stub_validates_shape(self):
        ball = Ball([1.0, 2.0], 0.5)
        self.assertEqual(ball.ndim, 2)
        np.testing.assert_allclose(ball.center, [1.0, 2.0])
        self.assertEqual(ball.radius, 0.5)

        with self.assertRaisesRegex(ValueError, "radius"):
            Ball([1.0, 2.0], 0.0)

        with self.assertRaisesRegex(NotImplementedError, "Ball compatibility"):
            Ball([1.0, 2.0, 3.0, 4.0], 1.0)

    def test_2d_tm_cuboid_smoothing_matches_julia_fixture(self):
        fixture = _load_fixture(SUBPIXEL_2D_TM_V5_FIXTURE)
        epsilon_xx = mesti_subpixel_smoothing(
            float(_scalar(fixture, "rect_delta_x")),
            _cuboid_from_fixture(fixture, "rect_domain"),
            _scalar(fixture, "rect_domain_epsilon"),
            [_cuboid_from_fixture(fixture, "rect_object")],
            [_scalar(fixture, "rect_object_epsilon")],
            _string(fixture, "rect_yBC"),
            _string(fixture, "rect_zBC"),
        )

        np.testing.assert_allclose(
            epsilon_xx,
            np.asarray(fixture["rect_epsilon_xx"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_2d_tm_without_subpixel_smoothing_matches_julia_fixture(self):
        fixture = _load_fixture(SUBPIXEL_2D_TM_V5_FIXTURE)
        epsilon_xx = mesti_subpixel_smoothing(
            float(_scalar(fixture, "rect_delta_x")),
            _cuboid_from_fixture(fixture, "rect_domain"),
            _scalar(fixture, "rect_domain_epsilon"),
            [_cuboid_from_fixture(fixture, "rect_object")],
            [_scalar(fixture, "rect_object_epsilon")],
            _string(fixture, "rect_yBC"),
            _string(fixture, "rect_zBC"),
            without_sb=True,
        )

        np.testing.assert_allclose(
            epsilon_xx,
            np.asarray(fixture["rect_epsilon_xx_without_sb"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_2d_tm_periodic_image_matches_julia_fixture(self):
        fixture = _load_fixture(SUBPIXEL_2D_TM_V5_FIXTURE)
        epsilon_xx = mesti_subpixel_smoothing(
            float(_scalar(fixture, "periodic_delta_x")),
            _cuboid_from_fixture(fixture, "periodic_domain"),
            _scalar(fixture, "periodic_domain_epsilon"),
            [_cuboid_from_fixture(fixture, "periodic_object")],
            [_scalar(fixture, "periodic_object_epsilon")],
            _string(fixture, "periodic_yBC"),
            _string(fixture, "periodic_zBC"),
        )

        np.testing.assert_allclose(
            epsilon_xx,
            np.asarray(fixture["periodic_epsilon_xx"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_2d_te_cuboid_inverse_epsilon_matches_julia_fixture(self):
        fixture = _load_fixture(SUBPIXEL_2D_TM_V5_FIXTURE)
        inv_epsilon_yy, inv_epsilon_zz, inv_epsilon_yz = mesti_subpixel_smoothing(
            float(_scalar(fixture, "rect_delta_x")),
            _cuboid_from_fixture(fixture, "rect_domain"),
            _scalar(fixture, "rect_domain_epsilon"),
            [_cuboid_from_fixture(fixture, "rect_object")],
            [_scalar(fixture, "rect_object_epsilon")],
            _string(fixture, "rect_yBC"),
            _string(fixture, "rect_zBC"),
            use_2D_TM=False,
            use_2D_TE=True,
        )

        np.testing.assert_allclose(
            inv_epsilon_yy,
            np.asarray(fixture["rect_inv_epsilon_yy"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            inv_epsilon_zz,
            np.asarray(fixture["rect_inv_epsilon_zz"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            inv_epsilon_yz,
            np.asarray(fixture["rect_inv_epsilon_yz"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_2d_te_without_subpixel_smoothing_matches_julia_fixture(self):
        fixture = _load_fixture(SUBPIXEL_2D_TM_V5_FIXTURE)
        inv_epsilon_yy, inv_epsilon_zz, inv_epsilon_yz = mesti_subpixel_smoothing(
            float(_scalar(fixture, "rect_delta_x")),
            _cuboid_from_fixture(fixture, "rect_domain"),
            _scalar(fixture, "rect_domain_epsilon"),
            [_cuboid_from_fixture(fixture, "rect_object")],
            [_scalar(fixture, "rect_object_epsilon")],
            _string(fixture, "rect_yBC"),
            _string(fixture, "rect_zBC"),
            use_2D_TM=False,
            use_2D_TE=True,
            without_sb=True,
        )

        np.testing.assert_allclose(
            inv_epsilon_yy,
            np.asarray(fixture["rect_inv_epsilon_yy_without_sb"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            inv_epsilon_zz,
            np.asarray(fixture["rect_inv_epsilon_zz_without_sb"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            inv_epsilon_yz,
            np.asarray(fixture["rect_inv_epsilon_yz_without_sb"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_2d_combined_tm_te_return_matches_julia_fixture(self):
        fixture = _load_fixture(SUBPIXEL_2D_TM_V5_FIXTURE)
        epsilon_xx, inv_epsilon = mesti_subpixel_smoothing(
            float(_scalar(fixture, "rect_delta_x")),
            _cuboid_from_fixture(fixture, "rect_domain"),
            _scalar(fixture, "rect_domain_epsilon"),
            [_cuboid_from_fixture(fixture, "rect_object")],
            [_scalar(fixture, "rect_object_epsilon")],
            _string(fixture, "rect_yBC"),
            _string(fixture, "rect_zBC"),
            use_2D_TE=True,
        )

        np.testing.assert_allclose(
            epsilon_xx,
            np.asarray(fixture["rect_epsilon_xx"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )
        for actual, key in zip(inv_epsilon, ("rect_inv_epsilon_yy", "rect_inv_epsilon_zz", "rect_inv_epsilon_yz")):
            np.testing.assert_allclose(
                actual,
                np.asarray(fixture[key], dtype=np.complex128),
                rtol=1e-12,
                atol=1e-12,
            )

    def test_3d_cuboid_tensor_smoothing_matches_julia_fixture(self):
        fixture = _load_fixture(SUBPIXEL_3D_CUBOID_V7_FIXTURE)
        _assert_3d_cuboid_fixture_matches(self, fixture, "rect3d")

    def test_3d_cuboid_without_subpixel_smoothing_matches_julia_fixture(self):
        fixture = _load_fixture(SUBPIXEL_3D_CUBOID_V7_FIXTURE)
        _assert_3d_cuboid_fixture_matches(self, fixture, "rect3d", without_sb=True)

    def test_3d_cuboid_edge_corner_tensor_smoothing_matches_julia_fixture(self):
        fixture = _load_fixture(SUBPIXEL_3D_CUBOID_V7_FIXTURE)
        _assert_3d_cuboid_fixture_matches(self, fixture, "edge3d")

    def test_3d_cuboid_edge_corner_without_subpixel_smoothing_matches_julia_fixture(self):
        fixture = _load_fixture(SUBPIXEL_3D_CUBOID_V7_FIXTURE)
        _assert_3d_cuboid_fixture_matches(self, fixture, "edge3d", without_sb=True)

    def test_2d_ball_smoothing_matches_julia_fixture(self):
        fixture = _load_fixture(SUBPIXEL_BALL_B1_FIXTURE)
        epsilon_xx, inv_epsilon = mesti_subpixel_smoothing(
            float(_scalar(fixture, "ball2d_delta_x")),
            _cuboid_from_fixture(fixture, "ball2d_domain"),
            _scalar(fixture, "ball2d_domain_epsilon"),
            [_ball_from_fixture(fixture, "ball2d_object")],
            [_scalar(fixture, "ball2d_object_epsilon")],
            _string(fixture, "ball2d_yBC"),
            _string(fixture, "ball2d_zBC"),
            use_2D_TE=True,
        )

        np.testing.assert_allclose(
            epsilon_xx,
            np.asarray(fixture["ball2d_epsilon_xx"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )
        for actual, key in zip(inv_epsilon, ("ball2d_inv_epsilon_yy", "ball2d_inv_epsilon_zz", "ball2d_inv_epsilon_yz")):
            np.testing.assert_allclose(
                actual,
                np.asarray(fixture[key], dtype=np.complex128),
                rtol=1e-12,
                atol=1e-12,
            )

    def test_2d_ball_without_subpixel_smoothing_matches_julia_fixture(self):
        fixture = _load_fixture(SUBPIXEL_BALL_B1_FIXTURE)
        epsilon_xx, inv_epsilon = mesti_subpixel_smoothing(
            float(_scalar(fixture, "ball2d_delta_x")),
            _cuboid_from_fixture(fixture, "ball2d_domain"),
            _scalar(fixture, "ball2d_domain_epsilon"),
            [_ball_from_fixture(fixture, "ball2d_object")],
            [_scalar(fixture, "ball2d_object_epsilon")],
            _string(fixture, "ball2d_yBC"),
            _string(fixture, "ball2d_zBC"),
            use_2D_TE=True,
            without_sb=True,
        )

        np.testing.assert_allclose(
            epsilon_xx,
            np.asarray(fixture["ball2d_epsilon_xx_without_sb"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )
        for actual, key in zip(
            inv_epsilon,
            (
                "ball2d_inv_epsilon_yy_without_sb",
                "ball2d_inv_epsilon_zz_without_sb",
                "ball2d_inv_epsilon_yz_without_sb",
            ),
        ):
            np.testing.assert_allclose(
                actual,
                np.asarray(fixture[key], dtype=np.complex128),
                rtol=1e-12,
                atol=1e-12,
            )

    def test_3d_ball_tensor_smoothing_matches_julia_fixture(self):
        fixture = _load_fixture(SUBPIXEL_BALL_B1_FIXTURE)
        _assert_3d_ball_fixture_matches(self, fixture, "ball3d")

    def test_3d_ball_without_subpixel_smoothing_matches_julia_fixture(self):
        fixture = _load_fixture(SUBPIXEL_BALL_B1_FIXTURE)
        _assert_3d_ball_fixture_matches(self, fixture, "ball3d", without_sb=True)

    def test_unsupported_subpixel_paths_are_explicit(self):
        domain = Cuboid([1.0, 1.0], [2.0, 2.0])
        obj = Cuboid([1.0, 1.0], [1.0, 1.0])

        with self.assertRaisesRegex(ValueError, "use_2D_TM"):
            mesti_subpixel_smoothing(
                1.0,
                domain,
                1.0,
                [obj],
                [2.0],
                "PEC",
                "PEC",
                use_2D_TM=False,
                use_2D_TE=False,
            )

        with self.assertRaisesRegex(TypeError, "3D subpixel"):
            mesti_subpixel_smoothing(1.0, Cuboid([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]), 1.0, [], [], "PEC", "PEC")

        with self.assertRaisesRegex(NotImplementedError, "Cuboid objects"):
            mesti_subpixel_smoothing(1.0, domain, 1.0, [object()], [2.0], "PEC", "PEC")

        with self.assertRaisesRegex(ValueError, "same length"):
            mesti_subpixel_smoothing(1.0, domain, 1.0, [obj], [], "PEC", "PEC")

    def test_ball_domain_subpixel_path_is_explicit_unsupported(self):
        domain_2d = Cuboid([1.0, 1.0], [2.0, 2.0])
        ball_2d = Ball([1.0, 1.0], 0.5)

        with self.assertRaisesRegex(TypeError, "domain must be a Cuboid"):
            mesti_subpixel_smoothing(1.0, ball_2d, 1.0, [], [], "PEC", "PEC")

        with self.assertRaisesRegex(NotImplementedError, "2D or 3D"):
            mesti_subpixel_smoothing(1.0, domain_2d, 1.0, [object()], [2.0], "PEC", "PEC")


if __name__ == "__main__":
    unittest.main()
