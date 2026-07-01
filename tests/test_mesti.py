import unittest
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat

import mesti.cudss_backend as cudss_backend
from mesti import (
    Matrices,
    Opts,
    PML,
    Source_struct,
    Syst,
    mesti,
    mesti_build_fdfd_matrix,
    mesti_matrix_solver,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
MESTI_3D_DIRECT_FIXTURE = FIXTURE_DIR / "mesti_3d_direct_diagonal_pec.mat"
MESTI_3D_DIRECT_V5_FIXTURE = FIXTURE_DIR / "mesti_3d_direct_v5_boundaries.mat"
MESTI_3D_DIRECT_OFFDIAGONAL_V5_FIXTURE = FIXTURE_DIR / "mesti_3d_direct_offdiagonal_v5.mat"
MESTI_DIRECT_OPTIONS_V5_FIXTURE = FIXTURE_DIR / "mesti_direct_options_v5.mat"


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


def _syst():
    return Syst(
        epsilon_xx=np.ones((2, 2), dtype=np.complex128),
        wavelength=2 * np.pi,
        dx=1.0,
        yBC="PMC",
        zBC="PMC",
    )


def _syst_3d_from_fixture(fixture):
    return Syst(
        epsilon_xx=np.asarray(fixture["epsilon_xx"], dtype=np.complex128),
        epsilon_yy=np.asarray(fixture["epsilon_yy"], dtype=np.complex128),
        epsilon_zz=np.asarray(fixture["epsilon_zz"], dtype=np.complex128),
        wavelength=float(_scalar(fixture, "wavelength")),
        dx=float(_scalar(fixture, "dx")),
        xBC=_bc_value(fixture, "xBC"),
        yBC=_bc_value(fixture, "yBC"),
        zBC=_bc_value(fixture, "zBC"),
    )


def _pml_layers_from_v5_fixture(fixture, prefix):
    layers = []
    for direction in ("x", "y", "z"):
        for side, suffix in (("-", "low"), ("+", "high")):
            layers.append(
                PML(
                    int(_scalar(fixture, f"{prefix}_{direction}PML_{suffix}_npixels")),
                    direction=direction,
                    side=side,
                )
            )
    return layers


def _fixture_array_or_none(fixture, key):
    if key not in fixture:
        return None
    return np.asarray(fixture[key], dtype=np.complex128)


def _syst_3d_v5_from_fixture(fixture, prefix):
    return Syst(
        epsilon_xx=np.asarray(fixture[f"{prefix}_epsilon_xx"], dtype=np.complex128),
        epsilon_xy=_fixture_array_or_none(fixture, f"{prefix}_epsilon_xy"),
        epsilon_xz=_fixture_array_or_none(fixture, f"{prefix}_epsilon_xz"),
        epsilon_yx=_fixture_array_or_none(fixture, f"{prefix}_epsilon_yx"),
        epsilon_yy=np.asarray(fixture[f"{prefix}_epsilon_yy"], dtype=np.complex128),
        epsilon_yz=_fixture_array_or_none(fixture, f"{prefix}_epsilon_yz"),
        epsilon_zx=_fixture_array_or_none(fixture, f"{prefix}_epsilon_zx"),
        epsilon_zy=_fixture_array_or_none(fixture, f"{prefix}_epsilon_zy"),
        epsilon_zz=np.asarray(fixture[f"{prefix}_epsilon_zz"], dtype=np.complex128),
        wavelength=float(_scalar(fixture, f"{prefix}_wavelength")),
        dx=float(_scalar(fixture, f"{prefix}_dx")),
        xBC=_bc_value(fixture, f"{prefix}_xBC"),
        yBC=_bc_value(fixture, f"{prefix}_yBC"),
        zBC=_bc_value(fixture, f"{prefix}_zBC"),
        PML=_pml_layers_from_v5_fixture(fixture, prefix),
    )


def _component_shapes(fixture):
    return (
        np.asarray(fixture["epsilon_xx"]).shape,
        np.asarray(fixture["epsilon_yy"]).shape,
        np.asarray(fixture["epsilon_zz"]).shape,
    )


def _component_sizes(shapes):
    return tuple(int(np.prod(shape)) for shape in shapes)


def _rhs_source_structs_from_fixture(fixture):
    B = np.asarray(fixture["B"], dtype=np.complex128)
    shapes = _component_shapes(fixture)
    sizes = _component_sizes(shapes)
    start = 0
    sources = []
    for shape, size in zip(shapes, sizes):
        stop = start + size
        # Python 3D Source_struct.pos is zero-based and inclusive, unlike
        # Julia's one-based start-plus-width form used to generate the fixture.
        pos = np.array([0, 0, 0, shape[0] - 1, shape[1] - 1, shape[2] - 1], dtype=int)
        data = B[start:stop, :].reshape((*shape, B.shape[1]), order="F")
        sources.append(Source_struct(pos=[pos], data=[data]))
        start = stop
    return sources


def _projection_source_structs_from_fixture(fixture):
    C = np.asarray(fixture["C"], dtype=np.complex128)
    shapes = _component_shapes(fixture)
    sizes = _component_sizes(shapes)
    start = 0
    projections = []
    for size in sizes:
        stop = start + size
        # Source_struct.ind stays component-local and zero-based in Python.
        projections.append(
            Source_struct(
                ind=[np.arange(size, dtype=int)],
                data=[C[:, start:stop].T],
            )
        )
        start = stop
    return projections


class MestiWrapperTest(unittest.TestCase):
    def test_cudss_direct_wrapper_field_profile_matches_scipy_when_available(self):
        _skip_unless_cudss_available(self)
        syst = _syst()
        B = np.array([[1.0 + 0.25j], [2.0], [-0.5j], [1.5 - 0.25j]], dtype=np.complex128)

        Ex_cudss, info = mesti(syst, B, opts=Opts(solver="cudss", verbal=False))
        Ex_scipy, _ = mesti(syst, B, opts=Opts(solver="scipy", verbal=False))

        np.testing.assert_allclose(Ex_cudss, Ex_scipy, rtol=1e-11, atol=1e-11)
        self.assertEqual(info.opts.solver, "cudss")
        self.assertTrue(info.opts.return_field_profile)

    def test_dense_rhs_field_profile_matches_manual_solve(self):
        syst = _syst()
        B = np.array([[1], [2], [3], [4]], dtype=np.complex128)

        Ex, info = mesti(syst, B, opts=Opts(verbal=False))
        A, _, _, _ = mesti_build_fdfd_matrix(
            syst.epsilon_xx,
            1.0,
            syst.yBC,
            syst.zBC,
        )
        X, _ = mesti_matrix_solver(Matrices(A=A, B=B), Opts(verbal=False))

        self.assertEqual(Ex.shape, (2, 2, 1))
        np.testing.assert_allclose(Ex[:, :, 0], X.reshape((2, 2), order="F"))
        self.assertTrue(info.is_symmetric_A)

    def test_projection_matches_manual_solve(self):
        syst = _syst()
        B = np.array([[1], [0], [0], [1]], dtype=np.complex128)
        C = np.array([[1, 0, 0, 1]], dtype=np.complex128)

        S, _ = mesti(syst, B, C=C)
        A, _, _, _ = mesti_build_fdfd_matrix(syst.epsilon_xx, 1.0, syst.yBC, syst.zBC)
        expected, _ = mesti_matrix_solver(Matrices(A=A, B=B, C=C))

        np.testing.assert_allclose(S, expected)

    def test_positional_opts_overload_for_field_profile_matches_keyword_opts(self):
        syst = _syst()
        B = np.array([[1], [2], [3], [4]], dtype=np.complex128)

        Ex_positional, info_positional = mesti(
            syst,
            B,
            Opts(solver="scipy", verbal=False, prefactor=2.0),
        )
        Ex_keyword, info_keyword = mesti(
            syst,
            B,
            opts=Opts(solver="scipy", verbal=False, prefactor=2.0),
        )

        np.testing.assert_allclose(Ex_positional, Ex_keyword)
        self.assertTrue(info_positional.opts.return_field_profile)
        self.assertTrue(info_keyword.opts.return_field_profile)

    def test_positional_opts_overload_for_projection_matches_keyword_opts(self):
        syst = _syst()
        B = np.array([[1], [0], [0], [1]], dtype=np.complex128)
        C = np.array([[1, 0, 0, 1]], dtype=np.complex128)

        S_positional, info_positional = mesti(
            syst,
            B,
            C,
            Opts(solver="scipy", verbal=False, prefactor=-2j),
        )
        S_keyword, info_keyword = mesti(
            syst,
            B,
            C=C,
            opts=Opts(solver="scipy", verbal=False, prefactor=-2j),
        )

        np.testing.assert_allclose(S_positional, S_keyword)
        self.assertFalse(info_positional.opts.return_field_profile)
        self.assertFalse(info_keyword.opts.return_field_profile)

    def test_source_struct_ind_uses_zero_based_indices(self):
        syst = _syst()
        source = Source_struct(
            ind=[np.array([0, 3])],
            data=[np.array([[1.0], [2.0]], dtype=np.complex128)],
        )
        dense_B = np.array([[1.0], [0.0], [0.0], [2.0]], dtype=np.complex128)

        Ex, _ = mesti(syst, source)
        dense_Ex, _ = mesti(syst, dense_B)

        np.testing.assert_allclose(Ex, dense_Ex)

    def test_source_struct_pos_uses_column_major_order(self):
        syst = _syst()
        source = Source_struct(
            pos=[np.array([0, 0, 1, 1])],
            data=[np.array([[[1.0], [3.0]], [[2.0], [4.0]]], dtype=np.complex128)],
        )
        dense_B = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.complex128)

        Ex, _ = mesti(syst, source)
        dense_Ex, _ = mesti(syst, dense_B)

        np.testing.assert_allclose(Ex, dense_Ex)

    def test_3d_dense_rhs_field_profile_matches_julia_fixture(self):
        fixture = _load_fixture(MESTI_3D_DIRECT_FIXTURE)
        syst = _syst_3d_from_fixture(fixture)

        Ex, Ey, Ez, info = mesti(
            syst,
            np.asarray(fixture["B"], dtype=np.complex128),
            opts=Opts(solver="scipy", verbal=False),
        )

        np.testing.assert_allclose(Ex, np.asarray(fixture["field_Ex"], dtype=np.complex128), rtol=5e-10, atol=5e-10)
        np.testing.assert_allclose(Ey, np.asarray(fixture["field_Ey"], dtype=np.complex128), rtol=5e-10, atol=5e-10)
        np.testing.assert_allclose(Ez, np.asarray(fixture["field_Ez"], dtype=np.complex128), rtol=5e-10, atol=5e-10)
        self.assertTrue(info.opts.return_field_profile)
        self.assertEqual(info.is_symmetric_A, bool(_scalar(fixture, "is_symmetric_A")))

    def test_3d_dense_projection_with_D_matches_julia_fixture(self):
        fixture = _load_fixture(MESTI_3D_DIRECT_FIXTURE)
        syst = _syst_3d_from_fixture(fixture)

        S, info = mesti(
            syst,
            np.asarray(fixture["B"], dtype=np.complex128),
            C=np.asarray(fixture["C"], dtype=np.complex128),
            D=np.asarray(fixture["D"], dtype=np.complex128),
            opts=Opts(solver="scipy", verbal=False),
        )

        np.testing.assert_allclose(
            S,
            np.asarray(fixture["projection_with_D"], dtype=np.complex128),
            rtol=5e-10,
            atol=5e-10,
        )
        self.assertFalse(info.opts.return_field_profile)

    def test_3d_sparse_rhs_and_projection_match_dense_fixture(self):
        fixture = _load_fixture(MESTI_3D_DIRECT_FIXTURE)
        syst = _syst_3d_from_fixture(fixture)

        S, _ = mesti(
            syst,
            sparse.csc_matrix(np.asarray(fixture["B"], dtype=np.complex128)),
            C=sparse.csc_matrix(np.asarray(fixture["C"], dtype=np.complex128)),
            D=sparse.csc_matrix(np.asarray(fixture["D"], dtype=np.complex128)),
            opts=Opts(solver="scipy", verbal=False),
        )

        np.testing.assert_allclose(
            S,
            np.asarray(fixture["projection_with_D"], dtype=np.complex128),
            rtol=5e-10,
            atol=5e-10,
        )

    def test_3d_source_struct_rhs_matches_julia_fixture(self):
        fixture = _load_fixture(MESTI_3D_DIRECT_FIXTURE)
        syst = _syst_3d_from_fixture(fixture)

        Ex, Ey, Ez, info = mesti(
            syst,
            _rhs_source_structs_from_fixture(fixture),
            opts=Opts(solver="scipy", verbal=False),
        )

        np.testing.assert_allclose(Ex, np.asarray(fixture["field_struct_Ex"], dtype=np.complex128), rtol=5e-10, atol=5e-10)
        np.testing.assert_allclose(Ey, np.asarray(fixture["field_struct_Ey"], dtype=np.complex128), rtol=5e-10, atol=5e-10)
        np.testing.assert_allclose(Ez, np.asarray(fixture["field_struct_Ez"], dtype=np.complex128), rtol=5e-10, atol=5e-10)
        np.testing.assert_allclose(Ex, np.asarray(fixture["field_Ex"], dtype=np.complex128), rtol=5e-10, atol=5e-10)
        self.assertTrue(info.opts.return_field_profile)

    def test_3d_source_struct_projection_matches_julia_fixture(self):
        fixture = _load_fixture(MESTI_3D_DIRECT_FIXTURE)
        syst = _syst_3d_from_fixture(fixture)

        S, info = mesti(
            syst,
            _rhs_source_structs_from_fixture(fixture),
            C=_projection_source_structs_from_fixture(fixture),
            D=np.asarray(fixture["D"], dtype=np.complex128),
            opts=Opts(solver="scipy", verbal=False),
        )

        np.testing.assert_allclose(
            S,
            np.asarray(fixture["projection_struct_with_D"], dtype=np.complex128),
            rtol=5e-10,
            atol=5e-10,
        )
        np.testing.assert_allclose(
            S,
            np.asarray(fixture["projection_with_D"], dtype=np.complex128),
            rtol=5e-10,
            atol=5e-10,
        )
        self.assertFalse(info.opts.return_field_profile)

    def test_3d_v5_boundary_pml_and_bloch_direct_fields_match_julia(self):
        fixture = _load_fixture(MESTI_3D_DIRECT_V5_FIXTURE)

        for prefix in ("pml", "bloch", "mixed_bc"):
            with self.subTest(prefix=prefix):
                syst = _syst_3d_v5_from_fixture(fixture, prefix)
                Ex, Ey, Ez, info = mesti(
                    syst,
                    np.asarray(fixture[f"{prefix}_B"], dtype=np.complex128),
                    opts=Opts(solver="scipy", verbal=False),
                )

                np.testing.assert_allclose(
                    Ex,
                    np.asarray(fixture[f"{prefix}_field_Ex"], dtype=np.complex128),
                    rtol=5e-10,
                    atol=5e-10,
                )
                np.testing.assert_allclose(
                    Ey,
                    np.asarray(fixture[f"{prefix}_field_Ey"], dtype=np.complex128),
                    rtol=5e-10,
                    atol=5e-10,
                )
                np.testing.assert_allclose(
                    Ez,
                    np.asarray(fixture[f"{prefix}_field_Ez"], dtype=np.complex128),
                    rtol=5e-10,
                    atol=5e-10,
                )
                self.assertTrue(info.opts.return_field_profile)
                self.assertEqual(info.is_symmetric_A, bool(_scalar(fixture, f"{prefix}_is_symmetric_A")))
                n_total = (Ex.size + Ey.size + Ez.size) // Ex.shape[-1]
                expected_shape = tuple(np.asarray(fixture[f"{prefix}_A_shape"], dtype=int).reshape(-1))
                self.assertEqual(expected_shape, (n_total, n_total))

    def test_3d_off_diagonal_direct_fields_and_projections_match_julia(self):
        fixture = _load_fixture(MESTI_3D_DIRECT_OFFDIAGONAL_V5_FIXTURE)

        for prefix in ("hermitian", "lossy", "pml"):
            with self.subTest(prefix=prefix):
                syst = _syst_3d_v5_from_fixture(fixture, prefix)
                Ex, Ey, Ez, info = mesti(
                    syst,
                    np.asarray(fixture[f"{prefix}_B"], dtype=np.complex128),
                    opts=Opts(solver="scipy", verbal=False),
                )

                np.testing.assert_allclose(
                    Ex,
                    np.asarray(fixture[f"{prefix}_field_Ex"], dtype=np.complex128),
                    rtol=5e-10,
                    atol=5e-10,
                )
                np.testing.assert_allclose(
                    Ey,
                    np.asarray(fixture[f"{prefix}_field_Ey"], dtype=np.complex128),
                    rtol=5e-10,
                    atol=5e-10,
                )
                np.testing.assert_allclose(
                    Ez,
                    np.asarray(fixture[f"{prefix}_field_Ez"], dtype=np.complex128),
                    rtol=5e-10,
                    atol=5e-10,
                )
                self.assertTrue(info.opts.return_field_profile)
                self.assertEqual(info.is_symmetric_A, bool(_scalar(fixture, f"{prefix}_is_symmetric_A")))

                S, projection_info = mesti(
                    syst,
                    np.asarray(fixture[f"{prefix}_B"], dtype=np.complex128),
                    C=np.asarray(fixture[f"{prefix}_C"], dtype=np.complex128),
                    D=np.asarray(fixture[f"{prefix}_D"], dtype=np.complex128),
                    opts=Opts(solver="scipy", verbal=False),
                )
                np.testing.assert_allclose(
                    S,
                    np.asarray(fixture[f"{prefix}_projection_with_D"], dtype=np.complex128),
                    rtol=5e-10,
                    atol=5e-10,
                )
                self.assertFalse(projection_info.opts.return_field_profile)

    def test_2d_direct_option_surface_matches_julia_fixture(self):
        fixture = _load_fixture(MESTI_DIRECT_OPTIONS_V5_FIXTURE)
        syst = Syst(
            epsilon_xx=np.asarray(fixture["twod_epsilon_xx"], dtype=np.complex128),
            wavelength=float(_scalar(fixture, "twod_wavelength")),
            dx=float(_scalar(fixture, "twod_dx")),
            PML=PML(1, direction="all"),
            PML_type="SC-PML",
        )
        B = np.asarray(fixture["twod_B"], dtype=np.complex128)
        prefactor = _scalar(fixture, "twod_prefactor")

        Ex, info = mesti(
            syst,
            B,
            opts=Opts(
                solver="scipy",
                verbal=False,
                prefactor=prefactor,
                exclude_PML_in_field_profiles=True,
            ),
        )

        np.testing.assert_allclose(
            Ex,
            np.asarray(fixture["twod_field_trim"], dtype=np.complex128),
            rtol=5e-10,
            atol=5e-10,
        )
        self.assertTrue(info.opts.return_field_profile)
        self.assertTrue(info.opts.exclude_PML_in_field_profiles)
        self.assertEqual(info.yPML[0].npixels, int(_scalar(fixture, "twod_yPML_low_npixels")))
        self.assertEqual(info.zPML[1].npixels, int(_scalar(fixture, "twod_zPML_high_npixels")))
        self.assertEqual(info.is_symmetric_A, bool(_scalar(fixture, "twod_is_symmetric_A")))

        S, projection_info = mesti(
            syst,
            B,
            C="transpose(B)",
            D=np.asarray(fixture["twod_D"], dtype=np.complex128),
            opts=Opts(
                solver="scipy",
                verbal=False,
                prefactor=prefactor,
                exclude_PML_in_field_profiles=True,
            ),
        )
        np.testing.assert_allclose(
            S,
            np.asarray(fixture["twod_projection_transpose_B"], dtype=np.complex128),
            rtol=5e-10,
            atol=5e-10,
        )
        self.assertFalse(projection_info.opts.return_field_profile)
        self.assertIsNone(projection_info.opts.exclude_PML_in_field_profiles)

    def test_2d_direct_bloch_convenience_matches_julia_fixture(self):
        fixture = _load_fixture(MESTI_DIRECT_OPTIONS_V5_FIXTURE)
        syst = Syst(
            epsilon_xx=np.asarray(fixture["twod_bloch_epsilon_xx"], dtype=np.complex128),
            wavelength=float(_scalar(fixture, "twod_bloch_wavelength")),
            dx=float(_scalar(fixture, "twod_bloch_dx")),
            ky_B=_scalar(fixture, "twod_bloch_ky_B"),
            kz_B=_scalar(fixture, "twod_bloch_kz_B"),
        )

        Ex, info = mesti(
            syst,
            np.asarray(fixture["twod_bloch_B"], dtype=np.complex128),
            opts=Opts(solver="scipy", verbal=False),
        )

        np.testing.assert_allclose(
            Ex,
            np.asarray(fixture["twod_bloch_field"], dtype=np.complex128),
            rtol=5e-10,
            atol=5e-10,
        )
        self.assertEqual(info.is_symmetric_A, bool(_scalar(fixture, "twod_bloch_is_symmetric_A")))

    def test_3d_direct_sc_pml_prefactor_and_pml_exclusion_match_julia_fixture(self):
        fixture = _load_fixture(MESTI_DIRECT_OPTIONS_V5_FIXTURE)
        syst = Syst(
            epsilon_xx=np.asarray(fixture["threed_epsilon_xx"], dtype=np.complex128),
            epsilon_yy=np.asarray(fixture["threed_epsilon_yy"], dtype=np.complex128),
            epsilon_zz=np.asarray(fixture["threed_epsilon_zz"], dtype=np.complex128),
            wavelength=float(_scalar(fixture, "threed_wavelength")),
            dx=float(_scalar(fixture, "threed_dx")),
            xBC="periodic",
            yBC="periodic",
            zBC="periodic",
            PML=PML(1, direction="all"),
            PML_type="SC-PML",
        )

        Ex, Ey, Ez, info = mesti(
            syst,
            np.asarray(fixture["threed_B"], dtype=np.complex128),
            opts=Opts(
                solver="scipy",
                verbal=False,
                prefactor=_scalar(fixture, "threed_prefactor"),
                exclude_PML_in_field_profiles=True,
            ),
        )

        np.testing.assert_allclose(
            Ex,
            np.asarray(fixture["threed_field_Ex_trim"], dtype=np.complex128),
            rtol=5e-10,
            atol=5e-10,
        )
        np.testing.assert_allclose(
            Ey,
            np.asarray(fixture["threed_field_Ey_trim"], dtype=np.complex128),
            rtol=5e-10,
            atol=5e-10,
        )
        np.testing.assert_allclose(
            Ez,
            np.asarray(fixture["threed_field_Ez_trim"], dtype=np.complex128),
            rtol=5e-10,
            atol=5e-10,
        )
        self.assertEqual(info.xPML[0].npixels, int(_scalar(fixture, "threed_xPML_low_npixels")))
        self.assertEqual(info.yPML[1].npixels, int(_scalar(fixture, "threed_yPML_high_npixels")))
        self.assertEqual(info.zPML[0].npixels, int(_scalar(fixture, "threed_zPML_low_npixels")))
        self.assertEqual(info.is_symmetric_A, bool(_scalar(fixture, "threed_is_symmetric_A")))

    def test_3d_direct_defaults_to_pec_boundaries(self):
        fixture = _load_fixture(MESTI_3D_DIRECT_FIXTURE)
        syst = Syst(
            epsilon_xx=np.asarray(fixture["epsilon_xx"], dtype=np.complex128),
            epsilon_yy=np.asarray(fixture["epsilon_yy"], dtype=np.complex128),
            epsilon_zz=np.asarray(fixture["epsilon_zz"], dtype=np.complex128),
            wavelength=float(_scalar(fixture, "wavelength")),
            dx=float(_scalar(fixture, "dx")),
        )

        Ex, Ey, Ez, _ = mesti(
            syst,
            np.asarray(fixture["B"], dtype=np.complex128),
            opts=Opts(solver="scipy", verbal=False),
        )

        np.testing.assert_allclose(Ex, np.asarray(fixture["field_Ex"], dtype=np.complex128), rtol=5e-10, atol=5e-10)
        np.testing.assert_allclose(Ey, np.asarray(fixture["field_Ey"], dtype=np.complex128), rtol=5e-10, atol=5e-10)
        np.testing.assert_allclose(Ez, np.asarray(fixture["field_Ez"], dtype=np.complex128), rtol=5e-10, atol=5e-10)

    def test_3d_direct_bloch_convenience_matches_julia_fixture(self):
        fixture = _load_fixture(MESTI_3D_DIRECT_V5_FIXTURE)
        base = _syst_3d_v5_from_fixture(fixture, "bloch")
        syst = Syst(
            epsilon_xx=base.epsilon_xx,
            epsilon_yy=base.epsilon_yy,
            epsilon_zz=base.epsilon_zz,
            wavelength=base.wavelength,
            dx=base.dx,
            kx_B=base.xBC / (base.epsilon_xx.shape[0] * base.dx),
            ky_B=base.yBC / (base.epsilon_yy.shape[1] * base.dx),
            kz_B=base.zBC / (base.epsilon_zz.shape[2] * base.dx),
        )

        Ex, Ey, Ez, info = mesti(
            syst,
            np.asarray(fixture["bloch_B"], dtype=np.complex128),
            opts=Opts(solver="scipy", verbal=False),
        )

        np.testing.assert_allclose(Ex, np.asarray(fixture["bloch_field_Ex"], dtype=np.complex128), rtol=5e-10, atol=5e-10)
        np.testing.assert_allclose(Ey, np.asarray(fixture["bloch_field_Ey"], dtype=np.complex128), rtol=5e-10, atol=5e-10)
        np.testing.assert_allclose(Ez, np.asarray(fixture["bloch_field_Ez"], dtype=np.complex128), rtol=5e-10, atol=5e-10)
        self.assertEqual(info.is_symmetric_A, bool(_scalar(fixture, "bloch_is_symmetric_A")))

    def test_direct_option_validation_errors_are_explicit(self):
        B = np.ones((4, 1), dtype=np.complex128)

        with self.assertRaisesRegex(ValueError, "D must be None"):
            mesti(_syst(), B, D=np.zeros((1, 1), dtype=np.complex128), opts=Opts(solver="scipy", verbal=False))

        with self.assertRaisesRegex(ValueError, "transpose"):
            mesti(_syst(), B, C="B.'", opts=Opts(solver="scipy", verbal=False))

        with self.assertRaisesRegex(ValueError, "exclude_PML_in_field_profiles"):
            mesti(
                _syst(),
                B,
                opts=Opts(solver="scipy", verbal=False, exclude_PML_in_field_profiles=1),
            )

        invalid_pml_type = _syst()
        invalid_pml_type.PML_type = "complex-stretch"
        with self.assertRaisesRegex(ValueError, "PML_type"):
            mesti(invalid_pml_type, B, opts=Opts(solver="scipy", verbal=False))

        incompatible_bloch = _syst()
        incompatible_bloch.ky_B = 0.1
        with self.assertRaisesRegex(ValueError, "ky_B"):
            mesti(incompatible_bloch, B, opts=Opts(solver="scipy", verbal=False))

        missing_bloch_wave_number = _syst()
        missing_bloch_wave_number.yBC = "Bloch"
        with self.assertRaisesRegex(ValueError, "Bloch"):
            mesti(missing_bloch_wave_number, B, opts=Opts(solver="scipy", verbal=False))

    def test_3d_direct_requires_all_diagonal_components(self):
        epsilon = np.ones((2, 2, 2), dtype=np.complex128)
        syst = Syst(
            epsilon_xx=epsilon,
            wavelength=2 * np.pi,
            dx=1.0,
            xBC="periodic",
            yBC="periodic",
            zBC="periodic",
        )

        with self.assertRaises(ValueError):
            mesti(syst, np.zeros((1, 1), dtype=np.complex128), opts=Opts(solver="scipy", verbal=False))

    def test_3d_off_diagonal_tensor_shape_validation(self):
        epsilon = np.ones((2, 2, 2), dtype=np.complex128)
        syst = Syst(
            epsilon_xx=epsilon,
            epsilon_yy=epsilon,
            epsilon_zz=epsilon,
            epsilon_xy=np.ones((1, 2, 2), dtype=np.complex128),
            wavelength=2 * np.pi,
            dx=1.0,
            xBC="periodic",
            yBC="periodic",
            zBC="periodic",
        )

        with self.assertRaisesRegex(ValueError, "epsilon_xy"):
            mesti(syst, np.zeros((24, 1), dtype=np.complex128), opts=Opts(solver="scipy", verbal=False))


if __name__ == "__main__":
    unittest.main()
