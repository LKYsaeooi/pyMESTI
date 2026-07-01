import importlib.util
import io
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from mesti import (
    ICNTL_DEFAULT,
    Mumps,
    UnsupportedMumpsOperation,
    basic_mumps_solve_demo,
    dense_matrix,
    dense_rhs,
    display_icntl,
    finalize,
    get_rhs,
    get_schur_complement,
    get_sol,
    has_matrix,
    has_rhs,
    has_schur,
    invoke_mumps,
    is_matrix_assembled,
    is_rhs_dense,
    mumps_det,
    mumps_factorize,
    mumps_schur_complement,
    mumps_schur_complement_demo,
    mumps_schur_complement_inplace,
    mumps_select_inv,
    mumps_solve,
    mumps_solve_inplace,
    provide_rhs,
    set_icntl,
    set_job,
    set_save_dir,
    set_save_prefix,
    sparse_matrix,
    sparse_rhs,
    suppress_printing,
    toggle_null_pivot,
)


EXAMPLE_DIR = Path(__file__).parents[1] / "examples"


def _load_example_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module {filename}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class MumpsCompatibilityTest(unittest.TestCase):
    def test_mumps_solve_matrix_rhs_matches_numpy(self):
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
        B = np.array([[1.0 + 0.5j, 0.0], [0.0, 2.0 - 0.25j], [3.0, 0.5j]], dtype=np.complex128)

        X = mumps_solve(A, B)

        np.testing.assert_allclose(X, np.linalg.solve(A.toarray(), B), atol=1e-12)

    def test_mumps_object_reuses_factorization_and_tracks_rhs_solution(self):
        A = sparse.diags([2 + 0.5j, 3 - 0.25j, 4 + 0.75j], format="csc", dtype=np.complex128)
        rhs1 = np.array([[2], [3], [4]], dtype=np.complex128)
        rhs2 = np.array([[4], [6], [8]], dtype=np.complex128)
        mumps = mumps_factorize(A)

        X1 = mumps_solve(mumps, rhs1)
        X2 = mumps_solve(mumps, rhs2)

        self.assertIsNotNone(mumps.factor)
        self.assertTrue(has_matrix(mumps))
        self.assertTrue(has_rhs(mumps))
        np.testing.assert_allclose(X1, np.linalg.solve(A.toarray(), rhs1), atol=1e-12)
        np.testing.assert_allclose(X2, np.linalg.solve(A.toarray(), rhs2), atol=1e-12)
        np.testing.assert_allclose(get_rhs(mumps), rhs2, atol=1e-12)
        np.testing.assert_allclose(get_sol(mumps), X2, atol=1e-12)

    def test_mumps_solve_inplace_writes_output(self):
        A = np.array([[2.0, 0.0], [0.0, 4.0]], dtype=np.complex128)
        B = np.array([[2.0], [8.0]], dtype=np.complex128)
        out = np.empty_like(B)

        mumps_solve_inplace(out, A, B)

        np.testing.assert_allclose(out, np.array([[1.0], [2.0]], dtype=np.complex128), atol=1e-12)

    def test_det_schur_and_selected_inverse_helpers(self):
        A = sparse.csc_matrix(
            np.array(
                [
                    [5.0, 1.0, 0.5],
                    [1.0, 4.0, 0.25],
                    [0.5, 0.25, 3.0],
                ],
                dtype=np.complex128,
            )
        )

        det_a = mumps_det(A)
        schur = mumps_schur_complement(A, [0, 2])
        selected = mumps_select_inv(A, np.array([0, 2]), np.array([2, 0]))

        dense = A.toarray()
        rest = np.array([1])
        keep = np.array([0, 2])
        expected_schur = dense[np.ix_(keep, keep)] - dense[np.ix_(keep, rest)] @ np.linalg.solve(
            dense[np.ix_(rest, rest)],
            dense[np.ix_(rest, keep)],
        )
        inverse = np.linalg.inv(dense)

        self.assertAlmostEqual(det_a, np.linalg.det(dense))
        np.testing.assert_allclose(schur, expected_schur, atol=1e-12)
        self.assertEqual(selected.shape, A.shape)
        np.testing.assert_allclose(selected.toarray()[0, 2], inverse[0, 2], atol=1e-12)
        np.testing.assert_allclose(selected.toarray()[2, 0], inverse[2, 0], atol=1e-12)

    def test_schur_inplace_stores_retrievable_matrix(self):
        A = sparse.csc_matrix(np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.complex128))
        mumps = Mumps(A)

        mumps_schur_complement_inplace(mumps, [0])

        self.assertTrue(has_schur(mumps))
        np.testing.assert_allclose(get_schur_complement(mumps), np.array([[2.5]], dtype=np.complex128))

    def test_controls_and_predicates_are_one_based_like_julia_docs(self):
        mumps = Mumps(np.eye(2, dtype=np.complex128))

        self.assertEqual(mumps.icntl, list(ICNTL_DEFAULT))
        set_icntl(mumps, 4, 0)
        suppress_printing(mumps)
        dense_matrix(mumps)
        sparse_matrix(mumps)
        sparse_rhs(mumps)
        dense_rhs(mumps)
        toggle_null_pivot(mumps)
        set_job(mumps, 2)
        set_save_dir(mumps, "mumps-save-dir")
        set_save_prefix(mumps, "prefix")

        self.assertEqual(mumps.icntl[3], 1)
        self.assertTrue(is_matrix_assembled(mumps))
        self.assertTrue(is_rhs_dense(mumps))
        self.assertEqual(mumps.icntl[23], 1)
        self.assertEqual(mumps.job, 2)
        self.assertEqual(mumps.save_dir, "mumps-save-dir")
        self.assertIn("ICNTL settings", display_icntl(mumps))

    def test_finalize_blocks_later_state_mutation(self):
        mumps = Mumps(np.eye(2, dtype=np.complex128))

        finalize(mumps)

        with self.assertRaisesRegex(RuntimeError, "finalized"):
            provide_rhs(mumps, np.ones((2, 1), dtype=np.complex128))

    def test_raw_mumps_invocation_is_explicit_unsupported(self):
        mumps = Mumps(np.eye(2, dtype=np.complex128), np.ones((2, 1), dtype=np.complex128))

        with self.assertRaisesRegex(UnsupportedMumpsOperation, "Raw MUMPS C/MPI invocation"):
            invoke_mumps(mumps)

    def test_basic_mumps_solve_demo_matches_julia_demo_contract(self):
        for dtype, sparse_rhs_input in ((np.complex128, False), (np.complex64, True)):
            with self.subTest(dtype=np.dtype(dtype).name, sparse_rhs=sparse_rhs_input):
                result = basic_mumps_solve_demo(n=8, nrhs=3, dtype=dtype, sparse_rhs=sparse_rhs_input)
                rhs = result.rhs.toarray() if sparse.issparse(result.rhs) else result.rhs

                self.assertEqual(result.solution.shape, (8, 3))
                self.assertEqual(result.sparse_rhs, sparse_rhs_input)
                self.assertLessEqual(result.residual_norm, result.residual_tolerance)
                np.testing.assert_allclose(
                    result.matrix.astype(np.complex128) @ result.solution,
                    np.asarray(rhs, dtype=np.complex128),
                    atol=result.residual_tolerance,
                )

    def test_schur_complement_demo_matches_julia_demo_contract(self):
        for dtype in (np.complex128, np.complex64):
            with self.subTest(dtype=np.dtype(dtype).name):
                result = mumps_schur_complement_demo(m=5, n=2, dtype=dtype)

                self.assertEqual(result.schur_indices.tolist(), [5, 6])
                self.assertEqual(result.schur.shape, (2, 2))
                self.assertLessEqual(result.relative_error, result.relative_tolerance)
                np.testing.assert_allclose(
                    result.schur,
                    result.expected_schur,
                    atol=result.relative_tolerance * max(1.0, np.linalg.norm(result.expected_schur)),
                )

    def test_raw_mumps_demo_scripts_are_importable(self):
        basic = _load_example_module("basic_mumps_solve.py", "basic_mumps_solve_example")
        schur = _load_example_module("mumps_schur_complement.py", "mumps_schur_complement_example")

        basic_result = basic.run_basic_mumps_solve(n=6, nrhs=2)
        schur_result = schur.run_mumps_schur_complement(m=4, n=2)

        self.assertLessEqual(basic_result.residual_norm, basic_result.residual_tolerance)
        self.assertLessEqual(schur_result.relative_error, schur_result.relative_tolerance)

    def test_hybrid_mpi_script_is_explicit_unsupported(self):
        hybrid = _load_example_module("hybrid_mpi.py", "hybrid_mpi_example")

        self.assertIn("MPI/hybrid_mpi.jl", hybrid.hybrid_mpi_unsupported_reason())
        self.assertIn("mpi4py", " ".join(hybrid.hybrid_mpi_migration_notes()))
        with self.assertRaisesRegex(UnsupportedMumpsOperation, "MPI worker orchestration"):
            hybrid.run_hybrid_mpi()

        stream = io.StringIO()
        self.assertEqual(hybrid.main(stream=stream), 1)
        self.assertIn("unsupported", stream.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
