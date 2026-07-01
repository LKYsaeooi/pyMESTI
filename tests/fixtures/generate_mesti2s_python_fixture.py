"""Regenerate the Python 2D TM mesti2s regression fixture.

Run from ``Simulation/python``:

    conda run -n simu_scattering_light python tests\\fixtures\\generate_mesti2s_python_fixture.py

This fixture is intentionally Python-generated because a Julia runtime was not
available locally when Step 11 of the port was implemented. Keep it as a
baseline even after Julia parity fixtures are added.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from mesti import Opts, PML, Syst, channel_type, mesti2s, wavefront  # noqa: E402


FIXTURE_PATH = Path(__file__).with_name("mesti2s_2d_tm_python_fixture.json")


def _complex_payload(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values, dtype=np.complex128)
    return {
        "real": values.real.tolist(),
        "imag": values.imag.tolist(),
    }


def _fixture_system() -> tuple[Syst, np.ndarray, np.ndarray]:
    k0dx = 1.3
    epsilon_xx = np.array(
        [
            [1.00 + 0.00j, 1.05 + 0.02j, 1.08 + 0.00j],
            [1.03 + 0.01j, 1.10 + 0.00j, 1.12 + 0.03j],
            [0.98 + 0.00j, 1.02 + 0.01j, 1.07 + 0.02j],
            [1.04 + 0.02j, 1.01 + 0.00j, 1.09 + 0.01j],
        ],
        dtype=np.complex128,
    )
    v_low = np.array(
        [
            [1.0 + 0.0j, 0.25 - 0.5j],
            [0.0 + 0.5j, -0.75 + 0.25j],
            [-0.25 + 0.75j, 0.5 + 0.0j],
        ],
        dtype=np.complex128,
    )
    syst = Syst(
        epsilon_xx=epsilon_xx,
        epsilon_low=1.21,
        epsilon_high=1.44,
        wavelength=float(2 * np.pi / k0dx),
        dx=1.0,
        yBC="periodic",
        zPML=[PML(3)],
    )
    return syst, epsilon_xx, v_low


def build_fixture() -> dict:
    syst, epsilon_xx, v_low = _fixture_system()
    t, channels_t, info_t = mesti2s(
        syst,
        channel_type(side="low"),
        channel_type(side="high"),
        Opts(verbal=False),
    )
    field_profile, _, info_field = mesti2s(
        syst,
        wavefront(v_low=v_low),
        Opts(verbal=False),
    )

    return {
        "fixture_format": 1,
        "description": (
            "Python-generated regression fixture for the 2D TM MESTI port. "
            "Created because no Julia runtime was available locally on 2026-05-23."
        ),
        "generator": "Simulation/python/tests/fixtures/generate_mesti2s_python_fixture.py",
        "system": {
            "epsilon_xx": _complex_payload(epsilon_xx),
            "epsilon_low": syst.epsilon_low,
            "epsilon_high": syst.epsilon_high,
            "wavelength": syst.wavelength,
            "dx": syst.dx,
            "yBC": syst.yBC,
            "zPML_npixels": syst.zPML[0].npixels,
        },
        "input": {
            "v_low": _complex_payload(v_low),
        },
        "expected": {
            "t": _complex_payload(t),
            "field_profile": _complex_payload(field_profile),
            "low_N_prop": channels_t.low.N_prop,
            "high_N_prop": channels_t.high.N_prop,
            "low_ind_prop": np.asarray(channels_t.low.ind_prop, dtype=int).tolist(),
            "high_ind_prop": np.asarray(channels_t.high.ind_prop, dtype=int).tolist(),
            "low_kzdx_prop": _complex_payload(channels_t.low.kzdx_prop),
            "high_kzdx_prop": _complex_payload(channels_t.high.kzdx_prop),
            "t_shape": list(t.shape),
            "field_profile_shape": list(field_profile.shape),
            "return_field_profile_for_t": bool(info_t.opts.return_field_profile),
            "return_field_profile_for_wavefront": bool(info_field.opts.return_field_profile),
        },
    }


def main() -> int:
    payload = build_fixture()
    FIXTURE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
