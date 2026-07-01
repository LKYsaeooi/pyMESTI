import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from simu_test_2D_TM import build_parser, main


class SimuTest2DTMEntryPointTest(unittest.TestCase):
    def _write_input(self, root, *, epsilon_low=1.0, epsilon_high=1.0):
        savemat(
            root / "epsilon.mat",
            {
                "syst_eps": np.ones((1, 0), dtype=np.complex128),
                "region_resolution": np.array([[0.1]]),
                "Wbg": np.array([[0.633]]),
                "epsilon_low": np.array([[epsilon_low]]),
                "epsilon_high": np.array([[epsilon_high]]),
            },
        )

    def test_main_writes_transmission_and_field_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_input(root)

            rc = main(["--root", str(root)])

            self.assertEqual(rc, 0)
            t_data = loadmat(root / "py_TM_mscaepsilon.mat")
            ex_data = loadmat(root / "py_Ex_eigen_epsilon.mat")
            self.assertIn("t", t_data)
            self.assertIn("Ex", ex_data)
            self.assertEqual(t_data["t"].shape, (1, 1))
            self.assertEqual(ex_data["Ex"].shape[0], 1)

    def test_main_accepts_npz_input_with_output_name_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(
                root / "converted.npz",
                syst_eps=np.ones((1, 0), dtype=np.complex128),
                region_resolution=np.array([[0.1]]),
                epsilon_low=np.array([[1.0]]),
                epsilon_high=np.array([[1.0]]),
            )

            rc = main(
                [
                    "--root",
                    str(root),
                    "--input",
                    "converted.npz",
                    "--output-input-name",
                    "epsilon.mat",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue((root / "py_TM_mscaepsilon.mat").exists())
            self.assertTrue((root / "py_Ex_eigen_epsilon.mat").exists())

    def test_parser_accepts_single_precision_and_skip_field_flags(self):
        args = build_parser().parse_args(
            [
                "--root",
                ".",
                "--solver",
                "mumpspy",
                "--method",
                "APF",
                "--single-precision-mumps",
                "--skip-field",
            ]
        )

        self.assertTrue(args.single_precision_mumps)
        self.assertTrue(args.skip_field)

    def test_main_can_stop_after_transmission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_input(root)

            rc = main(["--root", str(root), "--skip-field"])

            self.assertEqual(rc, 0)
            self.assertTrue((root / "py_TM_mscaepsilon.mat").exists())
            self.assertFalse((root / "py_Ex_eigen_epsilon.mat").exists())

    def test_main_rejects_zero_low_side_propagating_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_input(root, epsilon_low=0.0, epsilon_high=1.0)

            with self.assertRaisesRegex(ValueError, "low side has no propagating channels"):
                main(["--root", str(root)])

            self.assertFalse((root / "py_TM_mscaepsilon.mat").exists())
            self.assertFalse((root / "py_Ex_eigen_epsilon.mat").exists())

    def test_main_rejects_zero_high_side_propagating_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_input(root, epsilon_low=1.0, epsilon_high=0.0)

            with self.assertRaisesRegex(ValueError, "high side has no propagating channels"):
                main(["--root", str(root)])

            self.assertFalse((root / "py_TM_mscaepsilon.mat").exists())
            self.assertFalse((root / "py_Ex_eigen_epsilon.mat").exists())


if __name__ == "__main__":
    unittest.main()
