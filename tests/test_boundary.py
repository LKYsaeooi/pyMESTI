import unittest

import numpy as np

from mesti import (
    PML,
    build_ave_x_Ex,
    build_ddx_E,
    convert_BC,
    convert_BC_to_transverse,
    mesti_set_PML_params,
)


class BoundaryHelperTest(unittest.TestCase):
    def test_convert_bc(self):
        self.assertEqual(convert_BC("pec", "y"), "PEC")
        self.assertEqual(convert_BC("PMCPEC", "y"), "PMCPEC")
        self.assertEqual(convert_BC("periodic", "y"), "periodic")
        self.assertEqual(convert_BC(0.25, "y"), 0.25)
        with self.assertRaises(ValueError):
            convert_BC("bloch", "y")

    def test_convert_bc_to_transverse(self):
        self.assertEqual(convert_BC_to_transverse("PEC", "x", "x"), "Neumann")
        self.assertEqual(convert_BC_to_transverse("PEC", "x", "y"), "Dirichlet")
        self.assertEqual(convert_BC_to_transverse("PMC", "x", "x"), "Dirichlet")
        self.assertEqual(convert_BC_to_transverse("PECPMC", "y", "x"), "DirichletNeumann")
        self.assertEqual(convert_BC_to_transverse("periodic", "x", "y"), "periodic")

    def test_periodic_derivative_and_average_without_pml(self):
        ddx, avg, s_E, s_H, ind = build_ddx_E(3, "periodic", [PML(0), PML(0)], "y")
        vector = np.array([1, 2, 3], dtype=np.complex128)

        np.testing.assert_allclose(ddx @ vector, np.array([1, 1, -2], dtype=np.complex128))
        np.testing.assert_allclose(avg @ vector, np.array([1.5, 2.5, 2.0], dtype=np.complex128))
        np.testing.assert_allclose(s_E, np.ones(3))
        np.testing.assert_allclose(s_H, np.ones(3))
        self.assertEqual(ind, [None, None])

    def test_pec_derivative_and_average_without_pml(self):
        ddx, avg, _, _, _ = build_ddx_E(3, "PEC", [PML(0), PML(0)], "y")
        vector = np.array([1, 2, 3], dtype=np.complex128)

        np.testing.assert_allclose(ddx @ vector, np.array([1, 1, 1, -3], dtype=np.complex128))
        np.testing.assert_allclose(avg @ vector, np.array([0.5, 1.5, 2.5, 1.5], dtype=np.complex128))

    def test_average_x_ex_for_periodic(self):
        avg = build_ave_x_Ex(3, "periodic", "x")
        vector = np.array([1, 2, 3], dtype=np.complex128)

        np.testing.assert_allclose(avg @ vector, np.array([1.5, 2.5, 2.0], dtype=np.complex128))

    def test_pml_default_parameters_and_stretching(self):
        pml = mesti_set_PML_params([PML(1), PML(0)], 2 * np.pi / 10, [1.0, 1.0], "z")
        self.assertEqual(pml[0].power_sigma, 3)
        self.assertEqual(pml[0].power_kappa, 3)
        self.assertEqual(pml[0].alpha_max_over_omega, 0)
        self.assertGreater(pml[0].sigma_max_over_omega, 0)
        self.assertGreaterEqual(pml[0].kappa_max, 1)

        _, _, s_E, s_H, ind = build_ddx_E(4, "PEC", pml, "z")
        self.assertEqual(ind[0].tolist(), [0])
        self.assertFalse(np.allclose(s_E, np.ones_like(s_E)))
        self.assertFalse(np.allclose(s_H, np.ones_like(s_H)))


if __name__ == "__main__":
    unittest.main()

