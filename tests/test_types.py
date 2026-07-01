import unittest


class TypeSmokeTest(unittest.TestCase):
    def test_import_public_types(self):
        from mesti import (
            Channels_one_sided,
            Channels_two_sided,
            Info,
            Matrices,
            Opts,
            PML,
            Side,
            Source_struct,
            Syst,
            channel_index,
            channel_type,
            wavefront,
        )

        self.assertIsNone(Syst().epsilon_xx)
        self.assertEqual(PML(7).npixels, 7)
        self.assertIsNone(channel_type().side)
        self.assertIsNone(channel_index().ind_low)
        self.assertIsNone(wavefront().v_low)
        self.assertIsNone(Source_struct().data)
        self.assertIsNone(Matrices().A)
        self.assertIsNone(Opts().solver)
        self.assertIsNone(Info().timing_total)
        self.assertIsNone(Side().N_prop)
        self.assertIsNone(Channels_one_sided().N_prop)
        self.assertIsNone(Channels_two_sided().low)


if __name__ == "__main__":
    unittest.main()

