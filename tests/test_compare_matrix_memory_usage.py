import csv
import io
import unittest

from tests import compare_matrix_memory_usage as diagnostic


class MatrixMemoryDiagnosticTest(unittest.TestCase):
    def test_solver_benchmark_specs_include_mumps_apf(self):
        names = [spec.name for spec in diagnostic._solver_benchmark_specs()]

        self.assertIn("scipy_fs", names)
        self.assertIn("mumps_apf", names)
        self.assertIn("cudss_fs", names)
        self.assertIn("cudss_apf", names)

    def test_solver_benchmark_csv_reports_timing_and_memory_columns(self):
        report = diagnostic.SolverBenchmarkReport(
            case="small",
            backend="scipy_fs",
            status="ok",
            reason="",
            shape="4x1",
            matrix_n=4,
            matrix_nnz=10,
            input_storage_bytes=512,
            result_storage_bytes=64,
            host_current_bytes=1024,
            host_peak_bytes=2048,
            gpu_free_before_bytes=None,
            gpu_free_after_bytes=None,
            gpu_memory_delta_bytes=None,
            timing_build_s=None,
            timing_analyze_s=0.1,
            timing_factorize_s=0.2,
            timing_solve_s=0.3,
            timing_total_s=0.6,
            max_abs_error=0.0,
            rel_error=0.0,
        )
        stream = io.StringIO()

        diagnostic._write_solver_csv([report], stream)

        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(rows[0]["backend"], "scipy_fs")
        self.assertEqual(rows[0]["timing_total_s"], "0.6")
        self.assertEqual(rows[0]["host_peak_bytes"], "2048")
        self.assertIn("gpu_memory_delta_bytes", rows[0])
        self.assertEqual(float(rows[0]["max_abs_error"]), 0.0)


if __name__ == "__main__":
    unittest.main()
