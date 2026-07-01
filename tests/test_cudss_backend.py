import unittest

import numpy as np
from scipy import sparse

from mesti import cudss_backend


class CudssBackendProbeTest(unittest.TestCase):
    def test_probe_reports_required_fields(self):
        probe = cudss_backend.probe_environment()
        data = probe.as_dict()

        self.assertIn("available", data)
        self.assertIn("binding_strategy", data)
        self.assertIn("cuda", data)
        self.assertIn("cudss", data)
        self.assertIn("device_array", data)
        self.assertIn("compiler", data)
        self.assertIn("python_binding", data["cudss"])
        self.assertIsInstance(data["available"], bool)
        self.assertIsInstance(data["cuda"]["gpu_count"], int)
        self.assertIsInstance(data["cudss"]["headers"], list)
        self.assertIsInstance(data["cudss"]["libraries"], list)
        self.assertIsInstance(data["device_array"]["candidates"], list)
        self.assertIsInstance(data["compiler"]["candidates"], list)

    def test_cudss_environment_available_or_skips_cleanly(self):
        probe = cudss_backend.probe_environment()
        if not probe.available:
            self.skipTest(probe.unavailable_reason or "cuDSS GPU environment is not available")

        data = probe.as_dict()
        self.assertGreaterEqual(data["cuda"]["gpu_count"], 1)
        self.assertTrue(data["cuda"]["device_names"])
        self.assertIn(data["binding_strategy"], {"nvmath-bindings", "compiled-extension"})
        if data["binding_strategy"] == "nvmath-bindings":
            self.assertEqual(data["cudss"]["python_binding"], "nvmath.bindings.cudss")
            self.assertTrue(data["python_packages"]["nvmath.bindings.cudss"]["importable"])
        else:
            self.assertTrue(data["cudss"]["libraries"])
            self.assertTrue(data["cudss"]["headers"])
        self.assertIn(data["device_array"]["selected"], {"cupy", "cuda-python", "compiled-extension"})

    def test_cudss_multithreading_library_path_is_optional(self):
        path = cudss_backend._cudss_multithreading_lib()
        if path is not None:
            self.assertTrue(path.endswith(("libcudss_mtlayer_gomp.so.0", "cudss_mtlayer_gomp.dll")))

    def test_cudss_solve_matches_numpy_for_small_complex_system(self):
        probe = cudss_backend.probe_environment()
        if not probe.available:
            self.skipTest(probe.unavailable_reason or "cuDSS GPU environment is not available")
        if probe.binding_strategy != "nvmath-bindings":
            self.skipTest("G2 cuDSS smoke requires nvmath.bindings.cudss")

        A_dense = np.array(
            [
                [3.0 + 1.0j, 1.0 - 0.5j],
                [0.25 + 0.2j, 2.0 - 0.75j],
            ],
            dtype=np.complex128,
        )
        B = np.array(
            [
                [1.0 + 2.0j, 0.5],
                [3.0 - 1.0j, -2.0j],
            ],
            dtype=np.complex128,
        )

        X = cudss_backend.cudss_solve(sparse.csc_matrix(A_dense), B, opts=None, info=None)

        self.assertEqual(X.dtype, np.dtype(np.complex128))
        self.assertEqual(X.shape, B.shape)
        np.testing.assert_allclose(X, np.linalg.solve(A_dense, B), rtol=1e-11, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
