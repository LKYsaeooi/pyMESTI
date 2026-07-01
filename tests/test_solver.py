import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.io import loadmat
from scipy import sparse

import mesti.solver as solver_module
from mesti import Matrices, Opts, mesti_matrix_solver


FIXTURE_DIR = Path(__file__).parent / "fixtures"
SOLVER_FG_V5_FIXTURE = FIXTURE_DIR / "solver_fg_v5.mat"
SOLVER_MUMPS_SINGLE_PRECISION_V6_FIXTURE = FIXTURE_DIR / "solver_mumps_single_precision_v6.mat"


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


def _vector(data, key, dtype=None):
    return np.asarray(data[key], dtype=dtype).reshape(-1)


def _skip_unless_cudss_available(testcase):
    probe = solver_module.cudss_backend.probe_environment()
    if not probe.available:
        testcase.skipTest(probe.unavailable_reason or "cuDSS GPU environment is not available")
    if probe.binding_strategy != "nvmath-bindings":
        testcase.skipTest("cuDSS solver tests require nvmath.bindings.cudss")


class SolverTest(unittest.TestCase):
    def test_sparse_solver_residual(self):
        rng = np.random.default_rng(1234)
        n = 64
        nrhs = 4
        A = sparse.eye(n, format="csc", dtype=np.complex128)
        A = A + sparse.random(n, n, density=1 / n, format="csc", random_state=1) * 0.05
        B = rng.normal(size=(n, nrhs)) + 1j * rng.normal(size=(n, nrhs))

        X, info = mesti_matrix_solver(Matrices(A=A, B=B), Opts(verbal=False))

        residual = np.linalg.norm(A @ X - B)
        self.assertLessEqual(residual, 1e-8)
        self.assertIn(info.opts.solver, {"mumps", "scipy"})
        self.assertIsNotNone(info.timing_total)
        self.assertIsNotNone(info.timing_solve)

    def test_scipy_solver_can_be_requested(self):
        A = sparse.eye(2, format="csc", dtype=np.complex128) * 2
        B = np.array([[2], [4]], dtype=np.complex128)

        X, info = mesti_matrix_solver(Matrices(A=A, B=B), Opts(solver="scipy", verbal=False))

        self.assertEqual(info.opts.solver, "scipy")
        np.testing.assert_allclose(X, np.array([[1], [2]], dtype=np.complex128))

    def test_unknown_solver_name_is_rejected(self):
        A = sparse.eye(2, format="csc", dtype=np.complex128)
        B = np.ones((2, 1), dtype=np.complex128)

        with self.assertRaisesRegex(ValueError, "opts.solver"):
            mesti_matrix_solver(Matrices(A=A, B=B), Opts(solver="not-a-solver"))

    def test_cudss_solver_unavailable_error_is_explicit(self):
        A = sparse.eye(2, format="csc", dtype=np.complex128)
        B = np.ones((2, 1), dtype=np.complex128)
        original_require = solver_module.cudss_backend.require_available

        def unavailable():
            raise RuntimeError("opts.solver='cudss' requires a usable cuDSS backend: synthetic missing backend")

        try:
            solver_module.cudss_backend.require_available = unavailable
            with self.assertRaisesRegex(RuntimeError, "opts\\.solver='cudss'.*synthetic missing backend"):
                mesti_matrix_solver(Matrices(A=A, B=B), Opts(solver="cudss", verbal=False))
        finally:
            solver_module.cudss_backend.require_available = original_require

    def test_cudss_solver_dispatch_can_be_monkeypatched(self):
        A = sparse.eye(2, format="csc", dtype=np.complex128)
        B = np.array([[10], [20]], dtype=np.complex128)
        C = np.array([[1, 1]], dtype=np.complex128)
        calls = []
        original_require = solver_module.cudss_backend.require_available
        original_solve = solver_module.cudss_backend.cudss_solve

        def fake_solve(A_arg, B_arg, opts_arg, info_arg):
            calls.append((A_arg.shape, B_arg.shape, opts_arg.solver, info_arg is not None))
            return np.array([[2], [3]], dtype=np.complex128)

        try:
            solver_module.cudss_backend.require_available = lambda: None
            solver_module.cudss_backend.cudss_solve = fake_solve
            S, info = mesti_matrix_solver(Matrices(A=A, B=B, C=C), Opts(solver="cudss", verbal=False))
        finally:
            solver_module.cudss_backend.cudss_solve = original_solve
            solver_module.cudss_backend.require_available = original_require

        self.assertEqual(calls, [((2, 2), (2, 1), "cudss", True)])
        self.assertEqual(info.opts.solver, "cudss")
        np.testing.assert_allclose(S, np.array([[5]], dtype=np.complex128))

    def test_cudss_apf_dispatch_can_be_monkeypatched(self):
        A = sparse.eye(2, format="csc", dtype=np.complex128)
        B = np.ones((2, 1), dtype=np.complex128)
        C = np.ones((1, 2), dtype=np.complex128)
        calls = []
        original_require = solver_module.cudss_backend.require_available
        original_apf = solver_module.cudss_backend.cudss_apf

        def fake_apf(A_arg, B_arg, C_arg, opts_arg, transpose_B_arg, info_arg):
            calls.append((A_arg.shape, B_arg.shape, C_arg.shape, opts_arg.method, transpose_B_arg, info_arg is not None))
            return np.array([[7]], dtype=np.complex128)

        try:
            solver_module.cudss_backend.require_available = lambda: None
            solver_module.cudss_backend.cudss_apf = fake_apf
            S, info = mesti_matrix_solver(
                Matrices(A=A, B=B, C=C),
                Opts(solver="cudss", method="APF", verbal=False),
            )
        finally:
            solver_module.cudss_backend.cudss_apf = original_apf
            solver_module.cudss_backend.require_available = original_require

        self.assertEqual(calls, [((2, 2), (2, 1), (1, 2), "APF", False, True)])
        self.assertEqual(info.opts.solver, "cudss")
        self.assertEqual(info.opts.method, "APF")
        np.testing.assert_allclose(S, np.array([[7]], dtype=np.complex128))

    def test_cudss_factorize_and_solve_matches_scipy_when_available(self):
        _skip_unless_cudss_available(self)
        A = sparse.csc_matrix(
            np.array(
                [
                    [4.0 + 0.5j, 0.75 - 0.2j, 0.0],
                    [0.25 + 0.1j, 3.5 + 0.25j, 1.0 - 0.4j],
                    [0.1, -0.5j, 2.75 + 0.3j],
                ],
                dtype=np.complex128,
            )
        )
        B = np.array(
            [
                [1.0 + 0.5j, 0.0],
                [0.0, 2.0 - 0.25j],
                [3.0, 0.5j],
            ],
            dtype=np.complex128,
        )

        X, info = mesti_matrix_solver(Matrices(A=A, B=B), Opts(solver="cudss", verbal=False))

        np.testing.assert_allclose(X, np.linalg.solve(A.toarray(), B), rtol=1e-11, atol=1e-11)
        self.assertEqual(info.opts.solver, "cudss")
        self.assertIsNotNone(info.timing_analyze)
        self.assertIsNotNone(info.timing_factorize)
        self.assertIsNotNone(info.timing_solve)

    def test_cudss_projected_solve_sparse_rhs_and_d_match_scipy_when_available(self):
        _skip_unless_cudss_available(self)
        A = sparse.csc_matrix(
            np.array(
                [
                    [3.0 + 0.5j, 1.0 - 0.25j, 0.0],
                    [0.5 + 0.1j, 4.0 + 0.2j, 0.75],
                    [0.0, -0.5j, 2.5 + 0.3j],
                ],
                dtype=np.complex128,
            )
        )
        B = sparse.csc_matrix(
            np.array(
                [
                    [1.0 + 0.5j, 0.0, 2.0],
                    [0.0, 2.0 - 0.25j, 0.0],
                    [3.0, 0.5j, -1.0j],
                ],
                dtype=np.complex128,
            )
        )
        C = sparse.csc_matrix(np.array([[1.0, 0.0, 0.5 - 0.25j], [0.25j, -1.0, 0.75]], dtype=np.complex128))
        D = np.array([[0.1j, -0.2, 0.0], [0.5, 0.0, -0.25j]], dtype=np.complex128)

        S_cudss, info = mesti_matrix_solver(Matrices(A=A, B=B, C=C, D=D), Opts(solver="cudss", verbal=False, nrhs=1))
        S_scipy, _ = mesti_matrix_solver(Matrices(A=A, B=B, C=C, D=D), Opts(solver="scipy", verbal=False, nrhs=1))

        np.testing.assert_allclose(S_cudss, S_scipy, rtol=1e-11, atol=1e-11)
        self.assertEqual(info.opts.solver, "cudss")

    def test_cudss_apf_matches_scipy_when_available(self):
        _skip_unless_cudss_available(self)
        A = sparse.csc_matrix(
            np.array(
                [
                    [4.0 + 0.5j, 1.0 - 0.2j, 0.0],
                    [0.25 + 0.1j, 3.0 + 0.3j, 0.5],
                    [0.1, -0.4j, 2.5 + 0.2j],
                ],
                dtype=np.complex128,
            )
        )
        B = sparse.csc_matrix(
            np.array(
                [
                    [1.0 + 0.2j, 0.0],
                    [0.0, 2.0 - 0.1j],
                    [3.0, 0.5j],
                ],
                dtype=np.complex128,
            )
        )
        C = sparse.csc_matrix(np.array([[1.0, 0.0, 0.5 - 0.2j], [0.25j, -1.0, 0.75]], dtype=np.complex128))

        S_cudss, info = mesti_matrix_solver(Matrices(A=A, B=B, C=C), Opts(solver="cudss", method="APF", verbal=False))
        expected = C.toarray() @ np.linalg.solve(A.toarray(), B.toarray())

        np.testing.assert_allclose(S_cudss, expected, rtol=1e-11, atol=1e-11)
        self.assertEqual(info.opts.solver, "cudss")
        self.assertEqual(info.opts.method, "APF")
        self.assertIsNotNone(info.timing_build)
        self.assertIsNotNone(info.timing_factorize)

    def test_cudss_apf_transpose_b_matches_scipy_when_available(self):
        _skip_unless_cudss_available(self)
        A = sparse.csc_matrix(
            np.array(
                [
                    [5.0 + 0.25j, 0.75 - 0.1j, 0.2],
                    [0.75 - 0.1j, 4.0 + 0.5j, -0.3j],
                    [0.2, -0.3j, 3.0 + 0.4j],
                ],
                dtype=np.complex128,
            )
        )
        B = sparse.csc_matrix(
            np.array(
                [
                    [1.0 + 0.25j, 0.5],
                    [0.0, 2.0 - 0.1j],
                    [3.0, -0.5j],
                ],
                dtype=np.complex128,
            )
        )

        S_cudss, info = mesti_matrix_solver(
            Matrices(A=A, B=B, C="transpose(B)"),
            Opts(solver="cudss", method="APF", verbal=False),
        )
        B_dense = B.toarray()
        expected = B_dense.T @ np.linalg.solve(A.toarray(), B_dense)

        np.testing.assert_allclose(S_cudss, expected, rtol=1e-11, atol=1e-11)
        self.assertEqual(info.opts.solver, "cudss")
        self.assertEqual(info.opts.method, "APF")

    def test_cudss_sparse_rhs_batches_before_dense_solve(self):
        A = sparse.eye(3, format="csc", dtype=np.complex128)
        B = sparse.csc_matrix(
            np.array(
                [
                    [1, 0, 2],
                    [0, 3, 0],
                    [4, 0, 5],
                ],
                dtype=np.complex128,
            )
        )
        calls = []
        original_require = solver_module.cudss_backend.require_available
        original_solve = solver_module.cudss_backend.cudss_solve

        def fake_solve(A_arg, B_arg, opts_arg, info_arg):
            calls.append((A_arg.shape, B_arg.shape, sparse.issparse(B_arg)))
            return B_arg.toarray() if sparse.issparse(B_arg) else np.asarray(B_arg, dtype=np.complex128)

        try:
            solver_module.cudss_backend.require_available = lambda: None
            solver_module.cudss_backend.cudss_solve = fake_solve
            X, info = mesti_matrix_solver(Matrices(A=A, B=B), Opts(solver="cudss", verbal=False, nrhs=1))
        finally:
            solver_module.cudss_backend.cudss_solve = original_solve
            solver_module.cudss_backend.require_available = original_require

        self.assertEqual(calls, [((3, 3), (3, 1), True), ((3, 3), (3, 1), True), ((3, 3), (3, 1), True)])
        self.assertEqual(info.opts.solver, "cudss")
        np.testing.assert_allclose(X, B.toarray())

    def test_cudss_optimization_options_reach_backend(self):
        A = sparse.eye(2, format="csc", dtype=np.complex128)
        B = np.array([[10], [20]], dtype=np.complex128)
        calls = []
        original_require = solver_module.cudss_backend.require_available
        original_solve = solver_module.cudss_backend.cudss_solve

        def fake_solve(A_arg, B_arg, opts_arg, info_arg):
            calls.append(
                (
                    opts_arg.cudss_use_single_precision,
                    opts_arg.cudss_use_hybrid_memory,
                    opts_arg.cudss_hybrid_device_memory_limit,
                    opts_arg.cudss_register_cuda_memory,
                    info_arg is not None,
                )
            )
            return np.asarray(B_arg, dtype=np.complex128)

        try:
            solver_module.cudss_backend.require_available = lambda: None
            solver_module.cudss_backend.cudss_solve = fake_solve
            X, info = mesti_matrix_solver(
                Matrices(A=A, B=B),
                Opts(
                    solver="cudss",
                    verbal=False,
                    cudss_use_single_precision=True,
                    cudss_use_hybrid_memory=True,
                    cudss_hybrid_device_memory_limit="128MiB",
                    cudss_register_cuda_memory=False,
                ),
            )
        finally:
            solver_module.cudss_backend.cudss_solve = original_solve
            solver_module.cudss_backend.require_available = original_require

        self.assertEqual(calls, [(True, True, "128MiB", False, True)])
        self.assertEqual(info.opts.solver, "cudss")
        np.testing.assert_allclose(X, B)

    def test_cudss_single_precision_factorize_and_solve_matches_scipy_when_available(self):
        _skip_unless_cudss_available(self)
        A = sparse.csc_matrix(
            np.array(
                [
                    [4.0 + 0.5j, 0.75 - 0.2j, 0.0],
                    [0.25 + 0.1j, 3.5 + 0.25j, 1.0 - 0.4j],
                    [0.1, -0.5j, 2.75 + 0.3j],
                ],
                dtype=np.complex128,
            )
        )
        B = np.array(
            [
                [1.0 + 0.5j, 0.0],
                [0.0, 2.0 - 0.25j],
                [3.0, 0.5j],
            ],
            dtype=np.complex128,
        )

        X, info = mesti_matrix_solver(
            Matrices(A=A, B=B),
            Opts(solver="cudss", verbal=False, cudss_use_single_precision=True),
        )

        np.testing.assert_allclose(X, np.linalg.solve(A.toarray(), B), rtol=5e-5, atol=5e-6)
        self.assertEqual(info.opts.solver, "cudss")
        self.assertTrue(info.opts.cudss_use_single_precision)

    def test_cudss_hybrid_memory_factorize_and_solve_matches_scipy_when_available(self):
        _skip_unless_cudss_available(self)
        A = sparse.csc_matrix(
            np.array(
                [
                    [3.0 + 0.5j, 1.0 - 0.25j, 0.0],
                    [0.5 + 0.1j, 4.0 + 0.2j, 0.75],
                    [0.0, -0.5j, 2.5 + 0.3j],
                ],
                dtype=np.complex128,
            )
        )
        B = np.array(
            [
                [1.0 + 0.5j, 0.0],
                [0.0, 2.0 - 0.25j],
                [3.0, 0.5j],
            ],
            dtype=np.complex128,
        )

        X, info = mesti_matrix_solver(
            Matrices(A=A, B=B),
            Opts(
                solver="cudss",
                verbal=False,
                cudss_use_hybrid_memory=True,
                cudss_hybrid_device_memory_limit="128MiB",
            ),
        )

        np.testing.assert_allclose(X, np.linalg.solve(A.toarray(), B), rtol=1e-11, atol=1e-11)
        self.assertEqual(info.opts.solver, "cudss")
        self.assertTrue(info.opts.cudss_use_hybrid_memory)

    def test_cudss_apf_single_precision_hybrid_memory_matches_scipy_when_available(self):
        _skip_unless_cudss_available(self)
        A = sparse.csc_matrix(
            np.array(
                [
                    [4.0 + 0.5j, 1.0 - 0.2j, 0.0],
                    [0.25 + 0.1j, 3.0 + 0.3j, 0.5],
                    [0.1, -0.4j, 2.5 + 0.2j],
                ],
                dtype=np.complex128,
            )
        )
        B = sparse.csc_matrix(
            np.array(
                [
                    [1.0 + 0.2j, 0.0],
                    [0.0, 2.0 - 0.1j],
                    [3.0, 0.5j],
                ],
                dtype=np.complex128,
            )
        )
        C = sparse.csc_matrix(np.array([[1.0, 0.0, 0.5 - 0.2j], [0.25j, -1.0, 0.75]], dtype=np.complex128))

        S_cudss, info = mesti_matrix_solver(
            Matrices(A=A, B=B, C=C),
            Opts(
                solver="cudss",
                method="APF",
                verbal=False,
                cudss_use_single_precision=True,
                cudss_use_hybrid_memory=True,
                cudss_hybrid_device_memory_limit="128MiB",
            ),
        )
        expected = C.toarray() @ np.linalg.solve(A.toarray(), B.toarray())

        np.testing.assert_allclose(S_cudss, expected, rtol=5e-5, atol=5e-6)
        self.assertEqual(info.opts.solver, "cudss")
        self.assertEqual(info.opts.method, "APF")
        self.assertTrue(info.opts.cudss_use_single_precision)
        self.assertTrue(info.opts.cudss_use_hybrid_memory)

    @unittest.skipUnless(
        importlib.util.find_spec("mumps") is not None or importlib.util.find_spec("mumpspy") is not None,
        "no Python MUMPS binding is installed",
    )
    def test_mumps_solver_residual_when_binding_available(self):
        A = sparse.csc_matrix(
            np.array(
                [
                    [4.0 + 0.5j, 1.0 - 0.25j, 0.0],
                    [0.5 + 0.1j, 3.0 + 0.2j, 1.0],
                    [0.0, -0.5j, 2.5 + 0.3j],
                ],
                dtype=np.complex128,
            )
        )
        B = sparse.csc_matrix(
            np.array(
                [
                    [1.0 + 0.5j, 0.0],
                    [0.0, 2.0 - 0.25j],
                    [3.0, 0.5j],
                ],
                dtype=np.complex128,
            )
        )

        X, info = mesti_matrix_solver(Matrices(A=A, B=B), Opts(solver="MUMPS", verbal=False, nrhs=1))

        self.assertEqual(info.opts.solver, "mumps")
        self.assertIsNotNone(info.timing_factorize)
        residual = np.linalg.norm(A @ X - B.toarray())
        self.assertLessEqual(residual, 1e-10)

    def test_projected_solution_and_d_subtraction(self):
        A = sparse.eye(3, format="csc", dtype=np.complex128) * 2
        B = np.array([[2], [4], [6]], dtype=np.complex128)
        C = np.array([[1, 0, 1]], dtype=np.complex128)
        D = np.array([[0.5]], dtype=np.complex128)

        S, _ = mesti_matrix_solver(Matrices(A=A, B=B, C=C, D=D))

        self.assertEqual(S.shape, (1, 1))
        self.assertAlmostEqual(S[0, 0], 3.5)

    def test_transpose_b_projection_uses_nonconjugating_transpose(self):
        A = sparse.diags([2 + 0.5j, 3 - 0.25j, 4 + 0.75j], format="csc", dtype=np.complex128)
        B = np.array(
            [
                [1 + 2j, 0.5 - 0.25j],
                [2 - 1j, -1.5 + 0.75j],
                [0.25 + 0.5j, 1 - 2j],
            ],
            dtype=np.complex128,
        )

        S, info = mesti_matrix_solver(Matrices(A=A, B=B, C="transpose(B)"), Opts(solver="scipy", verbal=False))

        expected = B.T @ np.linalg.solve(A.toarray(), B)
        conjugating_result = B.conj().T @ np.linalg.solve(A.toarray(), B)
        np.testing.assert_allclose(S, expected, atol=1e-12)
        self.assertGreater(np.linalg.norm(S - conjugating_result), 1e-6)
        self.assertEqual(info.opts.method, "factorize_and_solve")

    def test_solver_method_aliases_normalize_to_factorize_and_solve(self):
        A = sparse.eye(2, format="csc", dtype=np.complex128) * 2
        B = np.array([[2], [4]], dtype=np.complex128)
        C = np.array([[1, 1]], dtype=np.complex128)

        S, info = mesti_matrix_solver(Matrices(A=A, B=B, C=C), Opts(solver="scipy", method="fs", verbal=False))

        np.testing.assert_allclose(S, np.array([[3]], dtype=np.complex128))
        self.assertEqual(info.opts.method, "factorize_and_solve")

    def test_apf_requires_mumpspy_backend(self):
        A = sparse.eye(2, format="csc", dtype=np.complex128)
        B = np.ones((2, 1), dtype=np.complex128)
        C = np.ones((1, 2), dtype=np.complex128)

        with self.assertRaisesRegex(RuntimeError, "APF"):
            mesti_matrix_solver(Matrices(A=A, B=B, C=C), Opts(solver="scipy", method="APF", verbal=False))

    def test_scipy_fg_matches_julia_fixture(self):
        fixture = _load_fixture(SOLVER_FG_V5_FIXTURE)
        A = sparse.csc_matrix(np.asarray(fixture["A"], dtype=np.complex128))
        B = sparse.csc_matrix(np.asarray(fixture["B"], dtype=np.complex128))
        C = sparse.csc_matrix(np.asarray(fixture["C"], dtype=np.complex128))

        S, info = mesti_matrix_solver(Matrices(A=A, B=B, C=C), Opts(solver="scipy", method="FG", verbal=False))

        self.assertEqual(info.opts.solver, "scipy")
        self.assertEqual(info.opts.method, "C*inv(U)*inv(L)*B")
        np.testing.assert_allclose(S, np.asarray(fixture["S_fg"], dtype=np.complex128), rtol=5e-12, atol=5e-12)
        np.testing.assert_allclose(
            S,
            np.asarray(fixture["S_factorize_and_solve"], dtype=np.complex128),
            rtol=5e-12,
            atol=5e-12,
        )
        np.testing.assert_allclose(
            np.linalg.svd(S, compute_uv=False),
            _vector(fixture, "S_fg_singular_values", dtype=float),
            rtol=5e-12,
            atol=5e-12,
        )

    def test_scipy_fg_transpose_b_matches_julia_fixture(self):
        fixture = _load_fixture(SOLVER_FG_V5_FIXTURE)
        A = sparse.csc_matrix(np.asarray(fixture["A"], dtype=np.complex128))
        B = sparse.csc_matrix(np.asarray(fixture["B"], dtype=np.complex128))

        S, info = mesti_matrix_solver(
            Matrices(A=A, B=B, C="transpose(B)"),
            Opts(solver="scipy", method="C*inv(U)*inv(L)*B", verbal=False),
        )

        self.assertEqual(info.opts.method, "C*inv(U)*inv(L)*B")
        np.testing.assert_allclose(
            S,
            np.asarray(fixture["S_transpose_b_fg"], dtype=np.complex128),
            rtol=5e-12,
            atol=5e-12,
        )
        np.testing.assert_allclose(
            S,
            np.asarray(fixture["S_transpose_b_factorize_and_solve"], dtype=np.complex128),
            rtol=5e-12,
            atol=5e-12,
        )
        np.testing.assert_allclose(
            np.linalg.svd(S, compute_uv=False),
            _vector(fixture, "S_transpose_b_singular_values", dtype=float),
            rtol=5e-12,
            atol=5e-12,
        )

    def test_fg_method_is_scoped_to_scipy_backend(self):
        A = sparse.eye(2, format="csc", dtype=np.complex128)
        B = np.ones((2, 1), dtype=np.complex128)
        C = np.ones((1, 2), dtype=np.complex128)
        original_solver_backend = solver_module._solver_backend

        try:
            solver_module._solver_backend = lambda opts: "mumpspy"
            with self.assertRaisesRegex(NotImplementedError, "solver='scipy'"):
                mesti_matrix_solver(Matrices(A=A, B=B, C=C), Opts(method="FG", verbal=False))
        finally:
            solver_module._solver_backend = original_solver_backend

    def test_symmetrize_k_is_rejected_at_matrix_solver(self):
        A = sparse.eye(2, format="csc", dtype=np.complex128)
        B = np.ones((2, 1), dtype=np.complex128)
        C = np.ones((1, 2), dtype=np.complex128)

        with self.assertRaisesRegex(ValueError, "symmetrize_K"):
            mesti_matrix_solver(Matrices(A=A, B=B, C=C), Opts(solver="scipy", symmetrize_K=True, verbal=False))

    def test_unsupported_solver_options_are_explicit(self):
        A = sparse.eye(2, format="csc", dtype=np.complex128)
        B = np.ones((2, 1), dtype=np.complex128)
        cases = [
            (Opts(solver="scipy", use_single_precision_MUMPS=True), "use_single_precision_MUMPS"),
            (Opts(solver="scipy", analysis_only=True), "analysis_only"),
            (Opts(solver="scipy", store_ordering=True), "store_ordering"),
            (Opts(solver="scipy", use_given_ordering=True), "use_given_ordering"),
            (Opts(solver="scipy", ordering=np.array([0, 1])), "ordering"),
            (Opts(solver="scipy", iterative_refinement=True), "iterative_refinement"),
            (Opts(solver="scipy", use_BLR=True), "BLR"),
            (Opts(solver="scipy", threshold_BLR=1e-7), "BLR"),
            (Opts(solver="scipy", icntl_36=1), "icntl_36"),
            (Opts(solver="scipy", icntl_38=1), "icntl_38"),
            (Opts(solver="scipy", nthreads_OMP=2), "nthreads_OMP"),
            (Opts(solver="scipy", use_L0_threads=True), "use_L0_threads"),
            (Opts(solver="scipy", use_METIS=True), "use_METIS"),
            (Opts(solver="scipy", write_LU_factor_to_disk=True), "write_LU_factor_to_disk"),
            (Opts(solver="scipy", cudss_use_single_precision=True), "cudss"),
            (Opts(solver="scipy", cudss_use_hybrid_memory=True), "cudss"),
            (Opts(solver="scipy", cudss_hybrid_device_memory_limit="128MiB"), "cudss"),
            (Opts(solver="scipy", cudss_register_cuda_memory=False), "cudss"),
        ]

        for opts, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((NotImplementedError, ValueError), message):
                    mesti_matrix_solver(Matrices(A=A, B=B), opts)

    def test_python_mumps_only_controls_are_backend_scoped(self):
        solver_module._validate_solver_options(Opts(use_METIS=True), "python-mumps")
        solver_module._validate_solver_options(Opts(write_LU_factor_to_disk=True), "python-mumps")
        solver_module._validate_solver_options(Opts(use_single_precision_MUMPS=True), "mumpspy")

        with self.assertRaisesRegex(NotImplementedError, "python-mumps"):
            solver_module._validate_solver_options(Opts(use_METIS=True), "mumpspy")
        with self.assertRaisesRegex(NotImplementedError, "python-mumps"):
            solver_module._validate_solver_options(Opts(write_LU_factor_to_disk=True), "mumpspy")
        with self.assertRaisesRegex(NotImplementedError, "mumpspy"):
            solver_module._validate_solver_options(Opts(use_single_precision_MUMPS=True), "python-mumps")
        with self.assertRaisesRegex(NotImplementedError, "mumpspy"):
            solver_module._validate_solver_options(Opts(use_single_precision_MUMPS=True), "scipy")

    def test_cudss_controls_are_backend_scoped(self):
        solver_module._validate_solver_options(Opts(cudss_use_single_precision=True), "cudss")
        solver_module._validate_solver_options(Opts(cudss_use_hybrid_memory=True), "cudss")
        solver_module._validate_solver_options(Opts(cudss_hybrid_device_memory_limit="128MiB"), "cudss")
        solver_module._validate_solver_options(Opts(cudss_register_cuda_memory=False), "cudss")

        for backend in ("scipy", "mumpspy", "python-mumps"):
            for opts in (
                Opts(cudss_use_single_precision=True),
                Opts(cudss_use_hybrid_memory=True),
                Opts(cudss_hybrid_device_memory_limit="128MiB"),
                Opts(cudss_register_cuda_memory=False),
            ):
                with self.subTest(backend=backend, opts=opts):
                    with self.assertRaisesRegex(NotImplementedError, "cuDSS"):
                        solver_module._validate_solver_options(opts, backend)

    def test_advanced_mumps_controls_are_delegated_for_python_bindings(self):
        cases = [
            (Opts(analysis_only=True), "analysis_only"),
            (Opts(store_ordering=True), "store_ordering"),
            (Opts(use_given_ordering=True), "use_given_ordering"),
            (Opts(ordering=np.array([0, 1])), "ordering"),
            (Opts(iterative_refinement=True), "iterative_refinement"),
            (Opts(use_BLR=True), "BLR"),
            (Opts(threshold_BLR=1e-7), "BLR"),
            (Opts(icntl_36=1), "icntl_36"),
            (Opts(icntl_38=1), "icntl_38"),
            (Opts(nthreads_OMP=2), "nthreads_OMP"),
            (Opts(use_L0_threads=True), "use_L0_threads"),
        ]

        for backend in ("python-mumps", "mumpspy"):
            for opts, message in cases:
                with self.subTest(backend=backend, message=message):
                    with self.assertRaisesRegex(NotImplementedError, message):
                        solver_module._validate_solver_options(opts, backend)

    @unittest.skipUnless(importlib.util.find_spec("mumpspy") is not None, "mumpspy is not installed")
    def test_mumpspy_single_precision_uses_complex64_system(self):
        import mumpspy

        A = sparse.csc_matrix(
            np.array(
                [
                    [4.0 + 0.5j, 1.0 - 0.25j],
                    [0.5 + 0.1j, 3.0 + 0.2j],
                ],
                dtype=np.complex128,
            )
        )
        B = np.array([[1.0 + 0.5j], [2.0 - 0.25j]], dtype=np.complex128)
        systems = []
        original_solver = mumpspy.MumpsSolver

        class RecordingSolver(original_solver):
            def __init__(self, *args, **kwargs):
                systems.append(kwargs.get("system"))
                super().__init__(*args, **kwargs)

        try:
            mumpspy.MumpsSolver = RecordingSolver
            X, info = mesti_matrix_solver(
                Matrices(A=A, B=B),
                Opts(solver="mumpspy", use_single_precision_MUMPS=True, verbal=False),
            )
        finally:
            mumpspy.MumpsSolver = original_solver

        expected = np.linalg.solve(A.toarray(), B)
        self.assertEqual(systems, ["complex64"])
        self.assertEqual(info.opts.solver, "mumpspy")
        self.assertTrue(info.opts.use_single_precision_MUMPS)
        np.testing.assert_allclose(X, expected, rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("mumpspy") is not None, "mumpspy is not installed")
    def test_mumpspy_apf_matches_factorize_and_solve(self):
        A = sparse.csc_matrix(
            np.array(
                [
                    [4.0 + 0.5j, 0.75 - 0.2j, 0.0],
                    [0.25 + 0.1j, 3.5 + 0.25j, 1.0 - 0.4j],
                    [0.1, -0.5j, 2.75 + 0.3j],
                ],
                dtype=np.complex128,
            )
        )
        B = np.array(
            [
                [1.0 + 0.5j, 0.0],
                [0.0, 2.0 - 0.25j],
                [3.0, 0.5j],
            ],
            dtype=np.complex128,
        )
        C = np.array(
            [
                [1.0, 0.0, 0.5 - 0.25j],
                [0.25j, -1.0, 0.75],
            ],
            dtype=np.complex128,
        )

        S, info = mesti_matrix_solver(Matrices(A=A, B=B, C=C), Opts(solver="mumpspy", method="APF", verbal=False))

        expected = C @ np.linalg.solve(A.toarray(), B)
        np.testing.assert_allclose(S, expected, rtol=1e-10, atol=1e-10)
        self.assertEqual(info.opts.method, "APF")
        self.assertIsNotNone(info.timing_factorize)

    @unittest.skipUnless(importlib.util.find_spec("mumpspy") is not None, "mumpspy is not installed")
    def test_mumpspy_apf_single_precision_uses_complex64_system(self):
        import mumpspy

        A = sparse.csc_matrix(
            np.array(
                [
                    [4.0 + 0.5j, 0.75 - 0.2j],
                    [0.25 + 0.1j, 3.5 + 0.25j],
                ],
                dtype=np.complex128,
            )
        )
        B = np.array([[1.0 + 0.5j], [2.0 - 0.25j]], dtype=np.complex128)
        C = np.array([[0.25j, 1.0]], dtype=np.complex128)
        systems = []
        original_solver = mumpspy.MumpsSolver

        class RecordingSolver(original_solver):
            def __init__(self, *args, **kwargs):
                systems.append(kwargs.get("system"))
                super().__init__(*args, **kwargs)

        try:
            mumpspy.MumpsSolver = RecordingSolver
            S, info = mesti_matrix_solver(
                Matrices(A=A, B=B, C=C),
                Opts(solver="mumpspy", method="APF", use_single_precision_MUMPS=True, verbal=False),
            )
        finally:
            mumpspy.MumpsSolver = original_solver

        expected = C @ np.linalg.solve(A.toarray(), B)
        self.assertEqual(systems, ["complex64"])
        self.assertEqual(info.opts.method, "APF")
        self.assertTrue(info.opts.use_single_precision_MUMPS)
        np.testing.assert_allclose(S, expected, rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("mumpspy") is not None, "mumpspy is not installed")
    @unittest.skipUnless(SOLVER_MUMPS_SINGLE_PRECISION_V6_FIXTURE.exists(), "Julia MUMPS v6 fixture is not generated")
    def test_mumpspy_single_precision_matches_julia_mumps_fixture(self):
        fixture = _load_fixture(SOLVER_MUMPS_SINGLE_PRECISION_V6_FIXTURE)
        A = sparse.csc_matrix(np.asarray(fixture["A"], dtype=np.complex128))
        B = sparse.csc_matrix(np.asarray(fixture["B"], dtype=np.complex128))
        C = sparse.csc_matrix(np.asarray(fixture["C"], dtype=np.complex128))
        opts = Opts(solver="mumpspy", use_single_precision_MUMPS=True, verbal=False, nrhs=2)

        X, info_x = mesti_matrix_solver(Matrices(A=A, B=B), opts)
        S, info_s = mesti_matrix_solver(Matrices(A=A, B=B, C=C), opts)

        self.assertEqual(info_x.opts.solver, "mumpspy")
        self.assertTrue(info_x.opts.use_single_precision_MUMPS)
        self.assertEqual(info_s.opts.method, "factorize_and_solve")
        self.assertEqual(fixture["X_single_output_eltype"], "ComplexF32")
        np.testing.assert_allclose(
            X,
            np.asarray(fixture["X_single_factorize_and_solve"], dtype=np.complex128),
            rtol=5e-5,
            atol=5e-6,
        )
        np.testing.assert_allclose(
            S,
            np.asarray(fixture["S_single_factorize_and_solve"], dtype=np.complex128),
            rtol=5e-5,
            atol=5e-6,
        )
        np.testing.assert_allclose(
            np.linalg.svd(X, compute_uv=False),
            _vector(fixture, "X_single_singular_values", dtype=float),
            rtol=5e-5,
            atol=5e-6,
        )
        self.assertLessEqual(float(np.asarray(fixture["X_single_vs_double_relerr"]).reshape(-1)[0]), 1e-4)
        self.assertLessEqual(
            float(np.asarray(fixture["S_factorize_and_solve_single_vs_double_relerr"]).reshape(-1)[0]),
            1e-4,
        )

    @unittest.skipUnless(importlib.util.find_spec("mumpspy") is not None, "mumpspy is not installed")
    @unittest.skipUnless(SOLVER_MUMPS_SINGLE_PRECISION_V6_FIXTURE.exists(), "Julia MUMPS v6 fixture is not generated")
    def test_mumpspy_apf_single_precision_matches_julia_mumps_fixture(self):
        fixture = _load_fixture(SOLVER_MUMPS_SINGLE_PRECISION_V6_FIXTURE)
        A = sparse.csc_matrix(np.asarray(fixture["A"], dtype=np.complex128))
        B = sparse.csc_matrix(np.asarray(fixture["B"], dtype=np.complex128))
        C = sparse.csc_matrix(np.asarray(fixture["C"], dtype=np.complex128))

        S, info = mesti_matrix_solver(
            Matrices(A=A, B=B, C=C),
            Opts(solver="mumpspy", method="APF", use_single_precision_MUMPS=True, verbal=False),
        )

        self.assertEqual(info.opts.method, "APF")
        self.assertTrue(info.opts.use_single_precision_MUMPS)
        self.assertEqual(fixture["S_single_apf_output_eltype"], "ComplexF32")
        np.testing.assert_allclose(
            S,
            np.asarray(fixture["S_single_apf"], dtype=np.complex128),
            rtol=5e-5,
            atol=5e-6,
        )
        np.testing.assert_allclose(
            np.linalg.svd(S, compute_uv=False),
            _vector(fixture, "S_single_apf_singular_values", dtype=float),
            rtol=5e-5,
            atol=5e-6,
        )
        self.assertLessEqual(float(np.asarray(fixture["S_apf_single_vs_double_relerr"]).reshape(-1)[0]), 1e-4)

    @unittest.skipUnless(importlib.util.find_spec("mumpspy") is not None, "mumpspy is not installed")
    def test_mumpspy_apf_transpose_b_uses_symmetric_augmented_matrix(self):
        A = sparse.csc_matrix(
            np.array(
                [
                    [4.0 + 0.5j, 0.75 - 0.2j, 0.1j],
                    [0.75 - 0.2j, 3.5 + 0.25j, 1.0 - 0.4j],
                    [0.1j, 1.0 - 0.4j, 2.75 + 0.3j],
                ],
                dtype=np.complex128,
            )
        )
        B = np.array(
            [
                [1.0 + 0.5j, 0.0],
                [0.0, 2.0 - 0.25j],
                [3.0, 0.5j],
            ],
            dtype=np.complex128,
        )

        S, _ = mesti_matrix_solver(
            Matrices(A=A, B=B, C="transpose(B)"),
            Opts(solver="mumpspy", method="APF", is_symmetric_A=True, verbal=False),
        )

        expected = B.T @ np.linalg.solve(A.toarray(), B)
        np.testing.assert_allclose(S, expected, rtol=1e-10, atol=1e-10)

    def test_sparse_rhs_projection_batches_without_densifying_full_rhs(self):
        A = sparse.diags([2, 3, 4, 5], format="csc", dtype=np.complex128)
        B = sparse.csc_matrix(
            np.array(
                [
                    [2, 0, 4],
                    [0, 6, 0],
                    [8, 0, 0],
                    [0, 10, 12],
                ],
                dtype=np.complex128,
            )
        )
        B_dense = B.toarray()
        B.toarray = lambda: (_ for _ in ()).throw(AssertionError("full sparse RHS was densified"))
        C = sparse.csc_matrix(
            np.array(
                [
                    [1, 0, 0, 1],
                    [0, 1, 1, 0],
                ],
                dtype=np.complex128,
            )
        )
        D = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.complex128)

        S, _ = mesti_matrix_solver(Matrices(A=A, B=B, C=C, D=D), Opts(verbal=False, nrhs=2))

        expected = C.toarray() @ (B_dense / np.array([2, 3, 4, 5], dtype=np.complex128)[:, np.newaxis]) - D
        np.testing.assert_allclose(S, expected, atol=1e-12)

    def test_sparse_rhs_default_batches_by_memory_budget(self):
        A = sparse.diags([2, 3, 4, 5, 6], format="csc", dtype=np.complex128)
        B = sparse.csc_matrix(
            np.array(
                [
                    [2, 0, 4, 0, 6],
                    [0, 6, 0, 8, 0],
                    [8, 0, 0, 0, 10],
                    [0, 10, 12, 0, 0],
                    [5, 0, 0, 15, 0],
                ],
                dtype=np.complex128,
            )
        )
        B_dense = B.toarray()
        C = np.array([[1, 0, 0, 1, 0], [0, 1, 1, 0, 1]], dtype=np.complex128)
        bytes_per_column = (
            A.shape[0]
            * np.dtype(np.complex128).itemsize
            * solver_module._DENSE_SOLVE_ARRAYS_PER_SPARSE_RHS_BATCH
        )
        toarray_shapes = []

        original_budget = solver_module._sparse_rhs_batch_memory_budget_bytes
        original_toarray = sparse.csc_matrix.toarray

        def recording_toarray(self, *args, **kwargs):
            toarray_shapes.append(self.shape)
            if self.shape == B.shape:
                raise AssertionError("full sparse RHS was densified")
            return original_toarray(self, *args, **kwargs)

        try:
            solver_module._sparse_rhs_batch_memory_budget_bytes = lambda: 2 * bytes_per_column
            sparse.csc_matrix.toarray = recording_toarray
            S, _ = mesti_matrix_solver(Matrices(A=A, B=B, C=C), Opts(solver="scipy", verbal=False))
        finally:
            sparse.csc_matrix.toarray = original_toarray
            solver_module._sparse_rhs_batch_memory_budget_bytes = original_budget

        expected = C @ (B_dense / np.array([2, 3, 4, 5, 6], dtype=np.complex128)[:, np.newaxis])
        np.testing.assert_allclose(S, expected, atol=1e-12)
        self.assertEqual(toarray_shapes, [(5, 2), (5, 2), (5, 1)])

    def test_explicit_nrhs_controls_sparse_rhs_batch_width(self):
        A = sparse.eye(5, format="csc", dtype=np.complex128)
        B = sparse.eye(5, format="csc", dtype=np.complex128)
        batch_calls = []

        original_rhs_slice = solver_module._rhs_slice

        def recording_rhs_slice(matrix, start, stop, *, dense):
            if sparse.issparse(matrix):
                batch_calls.append((start, stop, dense))
            return original_rhs_slice(matrix, start, stop, dense=dense)

        try:
            solver_module._rhs_slice = recording_rhs_slice
            X, _ = mesti_matrix_solver(Matrices(A=A, B=B), Opts(solver="scipy", verbal=False, nrhs=2))
        finally:
            solver_module._rhs_slice = original_rhs_slice

        np.testing.assert_allclose(X, B.toarray(), atol=1e-12)
        self.assertEqual(batch_calls, [(0, 2, True), (2, 4, True), (4, 5, True)])


if __name__ == "__main__":
    unittest.main()
