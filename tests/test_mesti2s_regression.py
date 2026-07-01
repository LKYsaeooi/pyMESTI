import json
import unittest
from pathlib import Path

import numpy as np

from mesti import Opts, PML, Syst, channel_type, mesti2s, wavefront


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mesti2s_2d_tm_python_fixture.json"


def _complex_array(payload):
    return np.asarray(payload["real"], dtype=np.float64) + 1j * np.asarray(
        payload["imag"],
        dtype=np.float64,
    )


def _load_fixture():
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _system_from_fixture(fixture):
    system = fixture["system"]
    return Syst(
        epsilon_xx=_complex_array(system["epsilon_xx"]),
        epsilon_low=system["epsilon_low"],
        epsilon_high=system["epsilon_high"],
        wavelength=system["wavelength"],
        dx=system["dx"],
        yBC=system["yBC"],
        zPML=[PML(system["zPML_npixels"])],
    )


class Mesti2SRegressionFixtureTest(unittest.TestCase):
    def test_python_fixture_low_to_high_transmission(self):
        fixture = _load_fixture()
        syst = _system_from_fixture(fixture)
        expected = fixture["expected"]

        t, channels, info = mesti2s(
            syst,
            channel_type(side="low"),
            channel_type(side="high"),
            Opts(verbal=False),
        )

        self.assertEqual(fixture["fixture_format"], 1)
        self.assertEqual(list(t.shape), expected["t_shape"])
        self.assertEqual(channels.low.N_prop, expected["low_N_prop"])
        self.assertEqual(channels.high.N_prop, expected["high_N_prop"])
        self.assertEqual(np.asarray(channels.low.ind_prop).tolist(), expected["low_ind_prop"])
        self.assertEqual(np.asarray(channels.high.ind_prop).tolist(), expected["high_ind_prop"])
        self.assertEqual(info.opts.return_field_profile, expected["return_field_profile_for_t"])
        np.testing.assert_allclose(channels.low.kzdx_prop, _complex_array(expected["low_kzdx_prop"]))
        np.testing.assert_allclose(channels.high.kzdx_prop, _complex_array(expected["high_kzdx_prop"]))
        np.testing.assert_allclose(t, _complex_array(expected["t"]), rtol=1e-10, atol=1e-10)

    def test_python_fixture_wavefront_field_profile(self):
        fixture = _load_fixture()
        syst = _system_from_fixture(fixture)
        expected = fixture["expected"]
        v_low = _complex_array(fixture["input"]["v_low"])

        field_profile, channels, info = mesti2s(
            syst,
            wavefront(v_low=v_low),
            Opts(verbal=False),
        )

        self.assertEqual(list(field_profile.shape), expected["field_profile_shape"])
        self.assertEqual(channels.low.N_prop, expected["low_N_prop"])
        self.assertEqual(info.opts.return_field_profile, expected["return_field_profile_for_wavefront"])
        np.testing.assert_allclose(
            field_profile,
            _complex_array(expected["field_profile"]),
            rtol=1e-10,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
