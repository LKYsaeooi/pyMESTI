"""Python entry point corresponding to ``simu_test_2D_TM.jl``."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from mesti import Opts, PML, Syst, channel_type, mesti2s, mesti_build_channels, wavefront


def _log(message: str) -> None:
    print(message, flush=True)


def _load_hdf5_mat(path: Path) -> dict[str, np.ndarray]:
    import h5py

    data: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        for key, value in handle.items():
            arr = np.asarray(value)
            if arr.dtype.fields and {"real", "imag"}.issubset(arr.dtype.fields):
                arr = arr["real"] + 1j * arr["imag"]
            if arr.ndim >= 2:
                # MATLAB v7.3 stores arrays in HDF5 dimension order; transpose
                # the 2D simulation inputs back to Julia/MATLAB (y, z) order.
                arr = arr.T
            data[key] = arr
    return data


def load_input(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    try:
        return {
            key: value
            for key, value in loadmat(path, squeeze_me=False).items()
            if not key.startswith("__")
        }
    except NotImplementedError:
        return _load_hdf5_mat(path)


def _scalar(data: dict[str, np.ndarray], key: str) -> float:
    if key not in data:
        raise KeyError(f"Input MAT file is missing key {key!r}.")
    arr = np.asarray(data[key])
    if arr.size != 1:
        raise ValueError(f"Input MAT key {key!r} must be scalar.")
    return float(np.real(arr.reshape(-1)[0]))


def _array(data: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Input MAT file is missing key {key!r}.")
    arr = np.asarray(data[key], dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError(f"Input MAT key {key!r} must be a 2D array.")
    return arr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute a 2D TM transmission matrix and open-channel field."
    )
    parser.add_argument("--root", type=Path, required=True, help="Folder containing epsilon.mat")
    parser.add_argument("--input", default="epsilon.mat", help="Input MAT filename")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output folder")
    parser.add_argument(
        "--output-input-name",
        default=None,
        help="Filename suffix to use for Julia-compatible output names when --input is a converted file.",
    )
    parser.add_argument(
        "--solver",
        default="auto",
        choices=["auto", "scipy", "MUMPS", "mumps", "mumpspy", "python-mumps", "cudss"],
        help="Linear solver backend. Auto uses an importable MUMPS binding, preferring mumpspy, before SciPy.",
    )
    parser.add_argument(
        "--nrhs",
        type=int,
        default=None,
        help="Override solver RHS batch width; omitted sparse RHS uses a memory-aware default.",
    )
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "APF", "FS", "factorize_and_solve"],
        help=(
            "Transmission solve method. Auto lets high-level mesti2s use mumpspy-backed APF "
            "when available; field-profile solves always use factorize-and-solve."
        ),
    )
    parser.add_argument(
        "--single-precision-mumps",
        action="store_true",
        help="Request opts.use_single_precision_MUMPS=True for supported MUMPS backends.",
    )
    parser.add_argument(
        "--cudss-use-single-precision",
        action="store_true",
        help="Request opts.cudss_use_single_precision=True for solver='cudss'.",
    )
    parser.add_argument(
        "--cudss-use-hybrid-memory",
        action="store_true",
        help="Request nvmath/cuDSS hybrid host+device memory mode for solver='cudss'.",
    )
    parser.add_argument(
        "--cudss-hybrid-device-memory-limit",
        default=None,
        help="Optional cuDSS hybrid device memory limit passed to nvmath, for example '2GiB'.",
    )
    parser.add_argument(
        "--cudss-no-register-cuda-memory",
        action="store_true",
        help="Set opts.cudss_register_cuda_memory=False for cuDSS hybrid memory mode.",
    )
    parser.add_argument(
        "--skip-field",
        action="store_true",
        help="Stop after saving the transmission matrix; useful for controlled production stress runs.",
    )
    return parser


def _build_opts(args: argparse.Namespace, method: str | None = None) -> Opts:
    return Opts(
        solver=args.solver,
        method=method,
        nrhs=args.nrhs,
        use_single_precision_MUMPS=True if args.single_precision_mumps else None,
        cudss_use_single_precision=True if args.cudss_use_single_precision else None,
        cudss_use_hybrid_memory=True if args.cudss_use_hybrid_memory else None,
        cudss_hybrid_device_memory_limit=args.cudss_hybrid_device_memory_limit,
        cudss_register_cuda_memory=False if args.cudss_no_register_cuda_memory else None,
    )


def _validate_open_channels(syst: Syst) -> None:
    channels = mesti_build_channels(syst)
    if channels.low.N_prop == 0:
        raise ValueError(
            "Cannot compute the open-channel field: the low side has no "
            "propagating channels for the input wavelength, dx, and epsilon_low."
        )
    if channels.high is not None and channels.high.N_prop == 0:
        raise ValueError(
            "Cannot compute the open-channel field: the high side has no "
            "propagating channels for the input wavelength, dx, and epsilon_high."
        )

class args_(object):
    def __init__(self, root, input_, output_dir):
        self.root = root
        self.input = input_
        self.output_dir = output_dir

def main(argv: list[str] | None = None) -> int:
    start_total = time.perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)
    
    output_dir = args.output_dir if args.output_dir is not None else args.root
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.root / args.input
    _log(f"Loading input {input_path}")
    data = load_input(input_path)

    syst = Syst(
        epsilon_xx=_array(data, "syst_eps"),
        epsilon_low=_scalar(data, "epsilon_low"),
        epsilon_high=_scalar(data, "epsilon_high"),
        length_unit="um",
        wavelength=0.633,
        dx=_scalar(data, "region_resolution"),
        yBC="periodic",
    )
    syst.zPML = [PML(int(round(syst.wavelength / syst.dx)))]
    _log(
        "Prepared system "
        f"epsilon_xx={syst.epsilon_xx.shape}, dx={syst.dx}, zPML={syst.zPML[0].npixels}, "
        f"solver={args.solver}, nrhs={args.nrhs}, single_precision_mumps={args.single_precision_mumps}, "
        f"cudss_single_precision={args.cudss_use_single_precision}, "
        f"cudss_hybrid_memory={args.cudss_use_hybrid_memory}, "
        f"cudss_hybrid_device_memory_limit={args.cudss_hybrid_device_memory_limit}"
    )
    _validate_open_channels(syst)

    _log("Computing low-to-high transmission matrix")
    start_t = time.perf_counter()
    t, channels, info_t = mesti2s(
        syst,
        channel_type(side="low"),
        channel_type(side="high"),
        _build_opts(args, None if args.method == "auto" else args.method),
    )
    output_input_name = args.output_input_name if args.output_input_name is not None else args.input
    savemat(output_dir / f"py_TM_msca{output_input_name}", {"t": t})
    _log(
        "Saved transmission "
        f"shape={t.shape}, elapsed={time.perf_counter() - start_t:.3f}s, "
        f"solver={info_t.opts.solver}, solve={info_t.timing_solve}"
    )

    if args.skip_field:
        _log("Skipping open-channel field profile because --skip-field was requested")
        _log(f"Done total_elapsed={time.perf_counter() - start_total:.3f}s")
        return 0

    _, _, vh = np.linalg.svd(t, full_matrices=True)
    v_low = np.zeros((channels.low.N_prop, 1), dtype=np.complex128)
    v_low[:, 0] = vh.conj().T[:, 0]
    _log("Computing open-channel field profile")
    start_ex = time.perf_counter()
    Ex, _, info_ex = mesti2s(
        syst,
        wavefront(v_low=v_low),
        _build_opts(args),
    )
    savemat(output_dir / f"py_Ex_eigen_{output_input_name}", {"Ex": Ex})
    _log(
        "Saved field profile "
        f"shape={Ex.shape}, elapsed={time.perf_counter() - start_ex:.3f}s, "
        f"solver={info_ex.opts.solver}, solve={info_ex.timing_solve}"
    )
    _log(f"Done total_elapsed={time.perf_counter() - start_total:.3f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
