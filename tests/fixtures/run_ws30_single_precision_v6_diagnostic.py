"""Run the v6 cropped-real single-precision MUMPS diagnostic.

Run from ``Simulation/python`` through WSL base Python, for example:

    python tests/fixtures/run_ws30_single_precision_v6_diagnostic.py transmission
    python tests/fixtures/run_ws30_single_precision_v6_diagnostic.py field

The script compares Python ``mumpspy`` with
``Opts(use_single_precision_MUMPS=True)`` against the existing Julia
double-MUMPS cropped-real ``Ws30 Ls7.5`` fixture.  It prints one
``JSON_RESULT`` line so the v6 handoff can record the measured metrics without
claiming production-size parity.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import resource
except ImportError:  # pragma: no cover - this diagnostic is intended for WSL.
    resource = None

from mesti import Opts, channel_type, mesti2s, wavefront
from tests.test_mesti2s_julia_regression import (
    WS30_CROPPED_FIXTURE,
    _load_fixture,
    _scalar,
    _system_from_fixture,
    _vector,
)


def _relerr(observed: np.ndarray, reference: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(reference)), np.finfo(float).eps)
    return float(np.linalg.norm(observed - reference) / denom)


def _base_payload(case: str, fixture: dict, elapsed: float, info) -> dict:
    epsilon = np.asarray(fixture["epsilon_xx"])
    return {
        "case": case,
        "fixture": str(WS30_CROPPED_FIXTURE),
        "epsilon_shape": list(epsilon.shape),
        "crop_name": str(_scalar(fixture, "crop_name")),
        "solver": info.opts.solver,
        "method": info.opts.method,
        "use_single_precision_MUMPS": bool(info.opts.use_single_precision_MUMPS),
        "nrhs": int(info.opts.nrhs),
        "return_field_profile": bool(info.opts.return_field_profile),
        "elapsed_wall_s": elapsed,
        "timing_total_s": None if info.timing_total is None else float(info.timing_total),
        "timing_build_s": None if info.timing_build is None else float(info.timing_build),
        "timing_factorize_s": None if info.timing_factorize is None else float(info.timing_factorize),
        "timing_solve_s": None if info.timing_solve is None else float(info.timing_solve),
        "peak_rss_kb": (
            None if resource is None else int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        ),
    }


def run_transmission(fixture: dict) -> dict:
    syst = _system_from_fixture(fixture)
    opts = Opts(solver="mumpspy", nrhs=95, verbal=False, use_single_precision_MUMPS=True)
    start = time.perf_counter()
    t, channels, info = mesti2s(syst, channel_type(side="low"), channel_type(side="high"), opts)
    elapsed = time.perf_counter() - start

    ref_t = np.asarray(fixture["t"], dtype=np.complex128)
    sv = np.linalg.svd(t, compute_uv=False)
    ref_sv = _vector(fixture, "singular_values", dtype=float)
    payload = _base_payload("ws30_center384_transmission_single_mumpspy", fixture, elapsed, info)
    payload.update(
        {
            "low_N_prop": int(channels.low.N_prop),
            "high_N_prop": int(channels.high.N_prop),
            "shape": list(t.shape),
            "t_norm": float(np.linalg.norm(t)),
            "ref_t_norm": float(np.linalg.norm(ref_t)),
            "t_relerr_vs_julia_double": _relerr(t, ref_t),
            "t_max_abs_diff_vs_julia_double": float(np.max(np.abs(t - ref_t))),
            "singular_relerr_vs_julia_double": _relerr(sv, ref_sv),
            "singular_max_abs_diff_vs_julia_double": float(np.max(np.abs(sv - ref_sv))),
            "top_singular_value": float(sv[0]),
            "top_singular_value_ref": float(ref_sv[0]),
        }
    )
    return payload


def run_field(fixture: dict) -> dict:
    syst = _system_from_fixture(fixture)
    opts = Opts(solver="mumpspy", nrhs=95, verbal=False, use_single_precision_MUMPS=True)
    start = time.perf_counter()
    field_profile, channels, info = mesti2s(
        syst,
        wavefront(v_low=np.asarray(fixture["v_low"], dtype=np.complex128)),
        opts,
    )
    elapsed = time.perf_counter() - start

    ref_field = np.asarray(fixture["field_profile"], dtype=np.complex128)
    payload = _base_payload("ws30_center384_field_single_mumpspy", fixture, elapsed, info)
    payload.update(
        {
            "low_N_prop": int(channels.low.N_prop),
            "high_N_prop": int(channels.high.N_prop),
            "shape": list(field_profile.shape),
            "field_norm": float(np.linalg.norm(field_profile)),
            "ref_field_norm": float(np.linalg.norm(ref_field)),
            "field_relerr_vs_julia_double": _relerr(field_profile, ref_field),
            "field_max_abs_diff_vs_julia_double": float(np.max(np.abs(field_profile - ref_field))),
        }
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("transmission", "field"))
    args = parser.parse_args()

    fixture = _load_fixture(WS30_CROPPED_FIXTURE)
    if args.mode == "transmission":
        payload = run_transmission(fixture)
    else:
        payload = run_field(fixture)
    print("JSON_RESULT " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
