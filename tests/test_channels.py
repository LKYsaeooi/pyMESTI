import unittest

import numpy as np

from mesti import (
    Channels_one_sided,
    Channels_two_sided,
    Syst,
    mesti_build_channels,
    mesti_build_transverse_function,
    mesti_setup_longitudinal,
)


class ChannelSetupTest(unittest.TestCase):
    def test_periodic_transverse_function_is_unitary(self):
        fun, kydx_all = mesti_build_transverse_function(5, "periodic")

        expected = np.array([-2, -1, 0, 1, 2], dtype=float) * (2 * np.pi / 5)
        np.testing.assert_allclose(kydx_all, expected)

        modes = fun(kydx_all)
        np.testing.assert_allclose(modes.conj().T @ modes, np.eye(5), atol=1e-12)

    def test_periodic_transverse_even_ordering_matches_julia(self):
        _, kydx_all = mesti_build_transverse_function(4, "periodic")

        expected = np.array([-1, 0, 1, 2], dtype=float) * (2 * np.pi / 4)
        np.testing.assert_allclose(kydx_all, expected)

    def test_setup_longitudinal_2d_periodic(self):
        kydx_all = np.array([-2 * np.pi / 3, 0.0, 2 * np.pi / 3])
        side = mesti_setup_longitudinal(
            k0dx=1.0,
            epsilon_bg=1.0,
            kxdx_all=None,
            kydx_all=kydx_all,
            kLambda_y=0,
            ind_zero_ky=1,
        )

        self.assertEqual(side.N_prop, 1)
        np.testing.assert_array_equal(side.ind_prop, np.array([1]))
        np.testing.assert_allclose(side.kydx_prop, np.array([0.0]))
        np.testing.assert_allclose(side.kzdx_prop, np.array([np.pi / 3]))
        np.testing.assert_allclose(side.sqrt_nu_prop, np.sqrt(np.sin(np.array([np.pi / 3]))))
        np.testing.assert_array_equal(side.ind_prop_conj, np.array([0]))

    def test_build_channels_2d_two_sided(self):
        channels = mesti_build_channels(3, "periodic", 1.0, 1.0, 1.0)

        self.assertIsInstance(channels, Channels_two_sided)
        self.assertIs(channels.low, channels.high)
        self.assertEqual(channels.low.N_prop, 1)
        np.testing.assert_allclose(channels.low.kydx_prop, np.array([0.0]))

    def test_build_channels_2d_one_sided(self):
        channels = mesti_build_channels(3, "periodic", 1.0, 1.0)

        self.assertIsInstance(channels, Channels_one_sided)
        self.assertEqual(channels.N_prop, 1)
        np.testing.assert_allclose(channels.kydx_prop, np.array([0.0]))

    def test_build_channels_from_syst(self):
        syst = Syst(
            epsilon_xx=np.ones((3, 0)),
            epsilon_low=1.0,
            epsilon_high=1.0,
            wavelength=2 * np.pi,
            dx=1.0,
            yBC="periodic",
        )

        channels = mesti_build_channels(syst)

        self.assertIsInstance(channels, Channels_two_sided)
        self.assertEqual(channels.low.N_prop, 1)


if __name__ == "__main__":
    unittest.main()

