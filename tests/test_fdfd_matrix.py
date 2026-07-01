import unittest
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat

from mesti import PML, mesti_build_fdfd_matrix


FIXTURE_DIR = Path(__file__).parent / "fixtures"
FDFD_3D_DIAGONAL_FIXTURE = FIXTURE_DIR / "fdfd_3d_diagonal_pec.mat"
FDFD_3D_DIAGONAL_V5_FIXTURE = FIXTURE_DIR / "fdfd_3d_diagonal_v5_boundaries.mat"
FDFD_3D_OFFDIAGONAL_V5_FIXTURE = FIXTURE_DIR / "fdfd_3d_offdiagonal_v5.mat"


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


def _pml_pair_from_fixture(data, prefix):
    return [
        PML(int(_scalar(data, f"{prefix}PML_low_npixels"))),
        PML(int(_scalar(data, f"{prefix}PML_high_npixels"))),
    ]


def _fixture_array_or_none(data, key):
    if key not in data:
        return None
    return np.asarray(data[key], dtype=np.complex128)


def _assert_3d_matrix_case_matches(testcase, fixture, prefix):
    A, is_symmetric, xPML, yPML, zPML = mesti_build_fdfd_matrix(
        np.asarray(fixture[f"{prefix}_epsilon_xx"], dtype=np.complex128),
        _scalar(fixture, f"{prefix}_k0dx"),
        _bc_value(fixture, f"{prefix}_yBC"),
        _bc_value(fixture, f"{prefix}_zBC"),
        yPML=_pml_pair_from_fixture(fixture, f"{prefix}_y"),
        zPML=_pml_pair_from_fixture(fixture, f"{prefix}_z"),
        use_UPML=bool(_scalar(fixture, f"{prefix}_use_UPML")),
        epsilon_yy=np.asarray(fixture[f"{prefix}_epsilon_yy"], dtype=np.complex128),
        epsilon_zz=np.asarray(fixture[f"{prefix}_epsilon_zz"], dtype=np.complex128),
        epsilon_xy=_fixture_array_or_none(fixture, f"{prefix}_epsilon_xy"),
        epsilon_xz=_fixture_array_or_none(fixture, f"{prefix}_epsilon_xz"),
        epsilon_yx=_fixture_array_or_none(fixture, f"{prefix}_epsilon_yx"),
        epsilon_yz=_fixture_array_or_none(fixture, f"{prefix}_epsilon_yz"),
        epsilon_zx=_fixture_array_or_none(fixture, f"{prefix}_epsilon_zx"),
        epsilon_zy=_fixture_array_or_none(fixture, f"{prefix}_epsilon_zy"),
        xBC=_bc_value(fixture, f"{prefix}_xBC"),
        xPML=_pml_pair_from_fixture(fixture, f"{prefix}_x"),
    )

    expected = np.asarray(fixture[f"{prefix}_A_dense"], dtype=np.complex128)
    diff = A.toarray() - expected
    testcase.assertTrue(sparse.isspmatrix_csc(A))
    testcase.assertEqual(A.shape, tuple(np.asarray(fixture[f"{prefix}_A_shape"], dtype=int).reshape(-1)))
    testcase.assertEqual(A.nnz, int(_scalar(fixture, f"{prefix}_A_nnz")))
    testcase.assertEqual(is_symmetric, bool(_scalar(fixture, f"{prefix}_is_symmetric_A")))
    testcase.assertEqual([layer.npixels for layer in xPML], [int(_scalar(fixture, f"{prefix}_xPML_low_npixels")), int(_scalar(fixture, f"{prefix}_xPML_high_npixels"))])
    testcase.assertEqual([layer.npixels for layer in yPML], [int(_scalar(fixture, f"{prefix}_yPML_low_npixels")), int(_scalar(fixture, f"{prefix}_yPML_high_npixels"))])
    testcase.assertEqual([layer.npixels for layer in zPML], [int(_scalar(fixture, f"{prefix}_zPML_low_npixels")), int(_scalar(fixture, f"{prefix}_zPML_high_npixels"))])
    testcase.assertLessEqual(float(np.max(np.abs(diff))), 2e-12)
    np.testing.assert_allclose(A.toarray(), expected, rtol=2e-12, atol=2e-12)


class FdfdMatrixTest(unittest.TestCase):
    def test_periodic_1d_laplacian_from_2d_tm_matrix(self):
        epsilon = np.zeros((3, 1), dtype=np.complex128)

        A, is_symmetric, _, _ = mesti_build_fdfd_matrix(
            epsilon,
            k0dx=0.0,
            yBC="periodic",
            zBC="periodic",
            yPML=[PML(0), PML(0)],
            zPML=[PML(0), PML(0)],
        )

        expected = np.array(
            [
                [2, -1, -1],
                [-1, 2, -1],
                [-1, -1, 2],
            ],
            dtype=np.complex128,
        )
        self.assertTrue(is_symmetric)
        np.testing.assert_allclose(A.toarray(), expected)

    def test_column_major_epsilon_diagonal_order(self):
        epsilon = np.array([[1, 3], [2, 4]], dtype=np.complex128)

        A, _, _, _ = mesti_build_fdfd_matrix(
            epsilon,
            k0dx=2.0,
            yBC="PMC",
            zBC="PMC",
            yPML=[PML(0), PML(0)],
            zPML=[PML(0), PML(0)],
        )
        no_material_A, _, _, _ = mesti_build_fdfd_matrix(
            np.zeros_like(epsilon),
            k0dx=2.0,
            yBC="PMC",
            zBC="PMC",
            yPML=[PML(0), PML(0)],
            zPML=[PML(0), PML(0)],
        )

        material_diag = -4.0 * epsilon.ravel(order="F")
        np.testing.assert_allclose((A - no_material_A).diagonal(), material_diag)

    def test_pml_upml_shape_and_returned_params(self):
        epsilon = np.ones((4, 3), dtype=np.complex128)

        A, is_symmetric, yPML, zPML = mesti_build_fdfd_matrix(
            epsilon,
            k0dx=2 * np.pi / 10,
            yBC="PEC",
            zBC="PEC",
            yPML=[PML(1), PML(0)],
            zPML=[PML(0), PML(1)],
            use_UPML=True,
        )

        self.assertEqual(A.shape, (12, 12))
        self.assertTrue(is_symmetric)
        self.assertIsNotNone(yPML[0].sigma_max_over_omega)
        self.assertIsNotNone(zPML[1].sigma_max_over_omega)
        self.assertTrue(sparse.isspmatrix_csc(A))

    def test_3d_diagonal_vectorial_matrix_matches_julia_fixture(self):
        fixture = _load_fixture(FDFD_3D_DIAGONAL_FIXTURE)

        A, is_symmetric, xPML, yPML, zPML = mesti_build_fdfd_matrix(
            np.asarray(fixture["epsilon_xx"], dtype=np.complex128),
            float(_scalar(fixture, "k0dx")),
            _string(fixture, "yBC"),
            _string(fixture, "zBC"),
            yPML=_pml_pair_from_fixture(fixture, "y"),
            zPML=_pml_pair_from_fixture(fixture, "z"),
            use_UPML=bool(_scalar(fixture, "use_UPML")),
            epsilon_yy=np.asarray(fixture["epsilon_yy"], dtype=np.complex128),
            epsilon_zz=np.asarray(fixture["epsilon_zz"], dtype=np.complex128),
            xBC=_string(fixture, "xBC"),
            xPML=_pml_pair_from_fixture(fixture, "x"),
        )

        self.assertTrue(sparse.isspmatrix_csc(A))
        self.assertEqual(A.shape, tuple(np.asarray(fixture["A_shape"], dtype=int).reshape(-1)))
        self.assertEqual(is_symmetric, bool(_scalar(fixture, "is_symmetric_A")))
        self.assertEqual([layer.npixels for layer in xPML], [0, 0])
        self.assertEqual([layer.npixels for layer in yPML], [0, 0])
        self.assertEqual([layer.npixels for layer in zPML], [0, 0])
        np.testing.assert_allclose(
            A.toarray(),
            np.asarray(fixture["A_dense"], dtype=np.complex128),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_3d_diagonal_v5_boundary_pml_and_bloch_fixtures_match_julia(self):
        fixture = _load_fixture(FDFD_3D_DIAGONAL_V5_FIXTURE)

        for prefix in ("pml", "bloch", "mixed_bc", "sc_pml"):
            with self.subTest(prefix=prefix):
                _assert_3d_matrix_case_matches(self, fixture, prefix)

    def test_3d_off_diagonal_v5_fixtures_match_julia(self):
        fixture = _load_fixture(FDFD_3D_OFFDIAGONAL_V5_FIXTURE)

        for prefix in ("hermitian", "lossy", "pml", "mixed_bc"):
            with self.subTest(prefix=prefix):
                _assert_3d_matrix_case_matches(self, fixture, prefix)

    def test_3d_off_diagonal_tensor_shape_validation(self):
        epsilon = np.ones((2, 2, 2), dtype=np.complex128)
        component_shapes = {
            "epsilon_xy": (1, 2, 2),
            "epsilon_xz": (2, 1, 2),
            "epsilon_yx": (2, 2, 1),
            "epsilon_yz": (1, 2, 2),
            "epsilon_zx": (2, 1, 2),
            "epsilon_zy": (2, 2, 1),
        }

        for name, shape in component_shapes.items():
            with self.subTest(name=name):
                kwargs = {name: np.ones(shape, dtype=np.complex128)}
                with self.assertRaisesRegex(ValueError, name):
                    mesti_build_fdfd_matrix(
                        epsilon,
                        0.7,
                        "periodic",
                        "periodic",
                        epsilon_yy=epsilon,
                        epsilon_zz=epsilon,
                        xBC="periodic",
                        **kwargs,
                    )


if __name__ == "__main__":
    unittest.main()
