"""Compare in-memory storage for MESTI matrix representations.

This is a standalone diagnostic script, not a pytest test case. It builds a
small 2D TM system, then reports matrix storage for:

- the FDFD operator A;
- dense, sparse, and Source_struct forms of B and C;
- dense and sparse D;
- the same objects after solver.py normalization.

Run from ``Simulation/python``:

    python tests/compare_matrix_memory_usage.py --ny 60 --nz 60 --nrhs 8
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
import time
import tracemalloc
from typing import Iterable, TextIO

import numpy as np
from scipy import sparse

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from mesti.fdfd_matrix import mesti_build_fdfd_matrix
from mesti.mesti import _source_like_to_matrix
import mesti.cudss_backend as cudss_backend
from mesti.solver import _as_csc, _as_rhs_matrix, _maybe_dense_or_sparse, mesti_matrix_solver
from mesti.types import Matrices, Opts, PML, Source_struct


@dataclass(frozen=True)
class MatrixReport:
    name: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    nnz: int | None
    storage_bytes: int
    dense_equiv_bytes: int
    density: float | None


@dataclass(frozen=True)
class SolverBenchmarkSpec:
    name: str
    solver: str
    method: str | None
    requires: str | None


@dataclass(frozen=True)
class SolverBenchmarkCase:
    name: str
    ny: int
    nz: int
    nrhs: int
    noutputs: int
    surface_z: int | None = None


@dataclass(frozen=True)
class SolverMatrixCase:
    name: str
    A: sparse.spmatrix
    B: sparse.spmatrix
    C: sparse.spmatrix
    n: int
    nnz: int
    input_storage_bytes: int


@dataclass(frozen=True)
class SolverBenchmarkReport:
    case: str
    backend: str
    status: str
    reason: str
    shape: str
    matrix_n: int
    matrix_nnz: int
    input_storage_bytes: int
    result_storage_bytes: int | None
    host_current_bytes: int | None
    host_peak_bytes: int | None
    gpu_free_before_bytes: int | None
    gpu_free_after_bytes: int | None
    gpu_memory_delta_bytes: int | None
    timing_build_s: float | None
    timing_analyze_s: float | None
    timing_factorize_s: float | None
    timing_solve_s: float | None
    timing_total_s: float | None
    max_abs_error: float | None
    rel_error: float | None


def _format_bytes(nbytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(nbytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{nbytes} B"


def _dense_nbytes(shape: tuple[int, ...], dtype: np.dtype | type = np.complex128) -> int:
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize


def _sparse_nbytes(matrix: sparse.spmatrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _report_matrix(name: str, matrix: object) -> MatrixReport:
    if sparse.issparse(matrix):
        shape = tuple(int(axis) for axis in matrix.shape)
        dense_equiv = _dense_nbytes(shape, matrix.dtype)
        entries = int(np.prod(shape, dtype=np.int64))
        density = (matrix.nnz / entries) if entries else 0.0
        return MatrixReport(
            name=name,
            kind=f"{type(matrix).__name__}/{matrix.getformat()}",
            shape=shape,
            dtype=str(matrix.dtype),
            nnz=int(matrix.nnz),
            storage_bytes=_sparse_nbytes(matrix),
            dense_equiv_bytes=dense_equiv,
            density=density,
        )

    array = np.asarray(matrix)
    shape = tuple(int(axis) for axis in array.shape)
    return MatrixReport(
        name=name,
        kind=type(matrix).__name__,
        shape=shape,
        dtype=str(array.dtype),
        nnz=None,
        storage_bytes=int(array.nbytes),
        dense_equiv_bytes=int(array.nbytes),
        density=1.0,
    )


def _report_virtual_dense(name: str, shape: tuple[int, ...], dtype: np.dtype | type = np.complex128) -> MatrixReport:
    nbytes = _dense_nbytes(shape, dtype)
    return MatrixReport(
        name=name,
        kind="virtual ndarray",
        shape=shape,
        dtype=str(np.dtype(dtype)),
        nnz=None,
        storage_bytes=nbytes,
        dense_equiv_bytes=nbytes,
        density=1.0,
    )


def _surface_indices(ny: int, z_index: int) -> np.ndarray:
    return np.arange(ny, dtype=int) + int(z_index) * ny


def _surface_values(ny: int, ncols: int, scale: float = 1.0) -> np.ndarray:
    y = np.arange(ny, dtype=float)[:, np.newaxis]
    col = np.arange(ncols, dtype=float)[np.newaxis, :]
    return scale * (1.0 + 0.01 * y + 0.1j * (col + 1.0))


def _dense_surface_matrix(total: int, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    matrix = np.zeros((total, values.shape[1]), dtype=np.complex128)
    matrix[indices, :] = values
    return matrix


def _sparse_surface_rhs(total: int, indices: np.ndarray, values: np.ndarray) -> sparse.csr_matrix:
    rows = np.tile(indices, values.shape[1])
    cols = np.repeat(np.arange(values.shape[1], dtype=int), len(indices))
    return sparse.coo_matrix(
        (values.reshape(-1, order="F"), (rows, cols)),
        shape=(total, values.shape[1]),
        dtype=np.complex128,
    ).tocsr()


def _dense_projection(total: int, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    matrix = np.zeros((values.shape[1], total), dtype=np.complex128)
    matrix[:, indices] = values.T
    return matrix


def _sparse_projection(total: int, indices: np.ndarray, values: np.ndarray) -> sparse.csr_matrix:
    rows = np.repeat(np.arange(values.shape[1], dtype=int), len(indices))
    cols = np.tile(indices, values.shape[1])
    return sparse.coo_matrix(
        (values.T.reshape(-1, order="C"), (rows, cols)),
        shape=(values.shape[1], total),
        dtype=np.complex128,
    ).tocsr()


def _make_epsilon(ny: int, nz: int) -> np.ndarray:
    y = np.linspace(0.0, 1.0, ny, dtype=float)[:, np.newaxis]
    z = np.linspace(0.0, 1.0, nz, dtype=float)[np.newaxis, :]
    return np.asarray(1.0 + 0.2 * y + 0.1 * z, dtype=np.complex128)


def _build_reports(args: argparse.Namespace) -> list[MatrixReport]:
    if args.noutputs is None:
        args.noutputs = args.nrhs
    if args.surface_z is None:
        args.surface_z = args.nz // 2
    if not 0 <= args.surface_z < args.nz:
        raise ValueError("--surface-z must be in [0, nz).")

    epsilon_xx = _make_epsilon(args.ny, args.nz)
    k0dx = (2 * np.pi / args.wavelength) * args.dx
    pml_pair_y = [PML(args.pml), PML(args.pml)]
    pml_pair_z = [PML(args.pml), PML(args.pml)]
    A, _, _, _ = mesti_build_fdfd_matrix(
        epsilon_xx,
        k0dx,
        args.ybc,
        args.zbc,
        yPML=pml_pair_y,
        zPML=pml_pair_z,
        use_UPML=not args.no_upml,
    )

    total = args.ny * args.nz
    indices = _surface_indices(args.ny, args.surface_z)
    b_values = _surface_values(args.ny, args.nrhs, scale=1.0)
    c_values = _surface_values(args.ny, args.noutputs, scale=0.5)

    B_dense_input = _dense_surface_matrix(total, indices, b_values)
    B_sparse_input = _sparse_surface_rhs(total, indices, b_values)
    B_source_input = Source_struct(
        pos=[np.array([0, args.surface_z, args.ny - 1, args.surface_z], dtype=int)],
        data=[b_values],
    )

    C_dense_input = _dense_projection(total, indices, c_values)
    C_sparse_input = _sparse_projection(total, indices, c_values)
    C_source_input = Source_struct(
        pos=[np.array([0, args.surface_z, args.ny - 1, args.surface_z], dtype=int)],
        data=[c_values],
    )

    D_dense_input = np.zeros((args.noutputs, args.nrhs), dtype=np.complex128)
    diag_count = min(args.noutputs, args.nrhs)
    D_dense_input[np.arange(diag_count), np.arange(diag_count)] = 1.0
    D_sparse_input = sparse.csr_matrix(D_dense_input)

    B_dense = _source_like_to_matrix(B_dense_input, (args.ny, args.nz), "B")
    B_sparse = _source_like_to_matrix(B_sparse_input, (args.ny, args.nz), "B")
    B_source = _source_like_to_matrix(B_source_input, (args.ny, args.nz), "B")
    C_dense = _source_like_to_matrix(C_dense_input, (args.ny, args.nz), "C", for_projection=True)
    C_sparse = _source_like_to_matrix(C_sparse_input, (args.ny, args.nz), "C", for_projection=True)
    C_source = _source_like_to_matrix(C_source_input, (args.ny, args.nz), "C", for_projection=True)

    reports = [
        _report_matrix("epsilon_xx dense material", epsilon_xx),
        _report_matrix("A built FDFD operator", A),
        _report_virtual_dense("A if materialized dense", A.shape, A.dtype),
        _report_matrix("B dense input", B_dense_input),
        _report_matrix("B sparse input CSR", B_sparse_input),
        _report_matrix("B dense wrapper", B_dense),
        _report_matrix("B sparse wrapper", B_sparse),
        _report_matrix("B Source_struct wrapper", B_source),
        _report_matrix("C dense input", C_dense_input),
        _report_matrix("C sparse input CSR", C_sparse_input),
        _report_matrix("C dense wrapper", C_dense),
        _report_matrix("C sparse wrapper", C_sparse),
        _report_matrix("C Source_struct wrapper", C_source),
        _report_matrix("D dense input", D_dense_input),
        _report_matrix("D sparse input CSR", D_sparse_input),
        _report_matrix("A solver-normalized", _as_csc(A, "A")),
        _report_matrix("B dense solver-normalized", _as_rhs_matrix(B_dense, "B")),
        _report_matrix("B sparse solver-normalized", _as_rhs_matrix(B_sparse, "B")),
        _report_matrix("B Source_struct solver-normalized", _as_rhs_matrix(B_source, "B")),
        _report_matrix("C dense solver-normalized", _maybe_dense_or_sparse(C_dense, "C")),
        _report_matrix("C sparse solver-normalized", _maybe_dense_or_sparse(C_sparse, "C")),
        _report_matrix("C Source_struct solver-normalized", _maybe_dense_or_sparse(C_source, "C")),
        _report_matrix("D dense solver-normalized", _maybe_dense_or_sparse(D_dense_input, "D")),
        _report_matrix("D sparse solver-normalized", _maybe_dense_or_sparse(D_sparse_input, "D")),
        _report_virtual_dense("X field result after solve", (total, args.nrhs)),
        _report_virtual_dense("S projected result after solve", (args.noutputs, args.nrhs)),
    ]
    return reports


def _solver_benchmark_specs() -> list[SolverBenchmarkSpec]:
    return [
        SolverBenchmarkSpec("scipy_fs", "scipy", None, None),
        SolverBenchmarkSpec("mumps_apf", "mumpspy", "APF", "mumpspy"),
        SolverBenchmarkSpec("cudss_fs", "cudss", None, "cudss"),
        SolverBenchmarkSpec("cudss_apf", "cudss", "APF", "cudss"),
    ]


def _build_solver_matrix_case(
    benchmark_case: SolverBenchmarkCase,
    *,
    wavelength: float = 10.0,
    dx: float = 1.0,
    pml: int = 0,
    ybc: str = "PEC",
    zbc: str = "PEC",
    no_upml: bool = False,
) -> SolverMatrixCase:
    surface_z = benchmark_case.surface_z if benchmark_case.surface_z is not None else benchmark_case.nz // 2
    if not 0 <= surface_z < benchmark_case.nz:
        raise ValueError("benchmark surface_z must be in [0, nz).")

    epsilon_xx = _make_epsilon(benchmark_case.ny, benchmark_case.nz)
    k0dx = (2 * np.pi / wavelength) * dx
    pml_pair_y = [PML(pml), PML(pml)]
    pml_pair_z = [PML(pml), PML(pml)]
    A, _, _, _ = mesti_build_fdfd_matrix(
        epsilon_xx,
        k0dx,
        ybc,
        zbc,
        yPML=pml_pair_y,
        zPML=pml_pair_z,
        use_UPML=not no_upml,
    )

    total = benchmark_case.ny * benchmark_case.nz
    indices = _surface_indices(benchmark_case.ny, surface_z)
    b_values = _surface_values(benchmark_case.ny, benchmark_case.nrhs, scale=1.0)
    c_values = _surface_values(benchmark_case.ny, benchmark_case.noutputs, scale=0.5)
    B = _sparse_surface_rhs(total, indices, b_values).tocsc()
    C = _sparse_projection(total, indices, c_values).tocsc()
    input_storage = _sparse_nbytes(A.tocsc()) + _sparse_nbytes(B) + _sparse_nbytes(C)
    return SolverMatrixCase(
        name=benchmark_case.name,
        A=A.tocsc(),
        B=B,
        C=C,
        n=int(A.shape[0]),
        nnz=int(A.nnz),
        input_storage_bytes=input_storage,
    )


def _parse_solver_benchmark_case(text: str) -> SolverBenchmarkCase:
    parts = text.split(":")
    if len(parts) != 5:
        raise ValueError("benchmark cases must use name:ny:nz:nrhs:noutputs.")
    name, ny, nz, nrhs, noutputs = parts
    values = [int(value) for value in (ny, nz, nrhs, noutputs)]
    if any(value <= 0 for value in values):
        raise ValueError("benchmark ny, nz, nrhs, and noutputs must be positive.")
    return SolverBenchmarkCase(name, *values)


def _gpu_memory_free_bytes() -> int | None:
    try:
        from cuda.bindings import runtime
    except ImportError:
        return None
    try:
        result = runtime.cudaMemGetInfo()
    except Exception:
        return None
    status = result[0]
    if status != runtime.cudaError_t.cudaSuccess:
        return None
    return int(result[1])


def _backend_unavailable_reason(spec: SolverBenchmarkSpec) -> str | None:
    if spec.requires == "mumpspy" and importlib.util.find_spec("mumpspy") is None:
        return "mumpspy is not installed"
    if spec.requires == "cudss":
        probe = cudss_backend.probe_environment()
        if not probe.available:
            return probe.unavailable_reason or "cuDSS GPU environment is not available"
        if probe.binding_strategy != "nvmath-bindings":
            return "cuDSS benchmark requires nvmath.bindings.cudss"
    return None


def _empty_solver_report(
    case: SolverMatrixCase,
    spec: SolverBenchmarkSpec,
    status: str,
    reason: str,
) -> SolverBenchmarkReport:
    return SolverBenchmarkReport(
        case=case.name,
        backend=spec.name,
        status=status,
        reason=reason,
        shape="",
        matrix_n=case.n,
        matrix_nnz=case.nnz,
        input_storage_bytes=case.input_storage_bytes,
        result_storage_bytes=None,
        host_current_bytes=None,
        host_peak_bytes=None,
        gpu_free_before_bytes=None,
        gpu_free_after_bytes=None,
        gpu_memory_delta_bytes=None,
        timing_build_s=None,
        timing_analyze_s=None,
        timing_factorize_s=None,
        timing_solve_s=None,
        timing_total_s=None,
        max_abs_error=None,
        rel_error=None,
    )


def _run_solver_once(
    case: SolverMatrixCase,
    spec: SolverBenchmarkSpec,
    baseline: np.ndarray | None,
) -> tuple[SolverBenchmarkReport, np.ndarray | None]:
    unavailable = _backend_unavailable_reason(spec)
    if unavailable is not None:
        return _empty_solver_report(case, spec, "unavailable", unavailable), None

    opts = Opts(solver=spec.solver, method=spec.method, verbal=False)
    gpu_free_before = _gpu_memory_free_bytes()
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result, info = mesti_matrix_solver(Matrices(A=case.A, B=case.B, C=case.C), opts)
        elapsed = time.perf_counter() - start
        host_current, host_peak = tracemalloc.get_traced_memory()
        gpu_free_after = _gpu_memory_free_bytes()
    except Exception as exc:
        elapsed = time.perf_counter() - start
        host_current, host_peak = tracemalloc.get_traced_memory()
        gpu_free_after = _gpu_memory_free_bytes()
        tracemalloc.stop()
        report = SolverBenchmarkReport(
            case=case.name,
            backend=spec.name,
            status="error",
            reason=f"{type(exc).__name__}: {exc}",
            shape="",
            matrix_n=case.n,
            matrix_nnz=case.nnz,
            input_storage_bytes=case.input_storage_bytes,
            result_storage_bytes=None,
            host_current_bytes=int(host_current),
            host_peak_bytes=int(host_peak),
            gpu_free_before_bytes=gpu_free_before,
            gpu_free_after_bytes=gpu_free_after,
            gpu_memory_delta_bytes=None if gpu_free_before is None or gpu_free_after is None else gpu_free_before - gpu_free_after,
            timing_build_s=None,
            timing_analyze_s=None,
            timing_factorize_s=None,
            timing_solve_s=None,
            timing_total_s=elapsed,
            max_abs_error=None,
            rel_error=None,
        )
        return report, None
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    max_abs_error: float | None = None
    rel_error: float | None = None
    if baseline is not None:
        diff = np.asarray(result) - baseline
        max_abs_error = float(np.max(np.abs(diff))) if diff.size else 0.0
        denom = float(np.max(np.abs(baseline))) if baseline.size else 0.0
        rel_error = max_abs_error / max(denom, 1.0)
    else:
        max_abs_error = 0.0
        rel_error = 0.0

    report = SolverBenchmarkReport(
        case=case.name,
        backend=spec.name,
        status="ok",
        reason="",
        shape="x".join(str(axis) for axis in np.asarray(result).shape),
        matrix_n=case.n,
        matrix_nnz=case.nnz,
        input_storage_bytes=case.input_storage_bytes,
        result_storage_bytes=int(np.asarray(result).nbytes),
        host_current_bytes=int(host_current),
        host_peak_bytes=int(host_peak),
        gpu_free_before_bytes=gpu_free_before,
        gpu_free_after_bytes=gpu_free_after,
        gpu_memory_delta_bytes=None if gpu_free_before is None or gpu_free_after is None else gpu_free_before - gpu_free_after,
        timing_build_s=getattr(info, "timing_build", None),
        timing_analyze_s=getattr(info, "timing_analyze", None),
        timing_factorize_s=getattr(info, "timing_factorize", None),
        timing_solve_s=getattr(info, "timing_solve", None),
        timing_total_s=elapsed,
        max_abs_error=max_abs_error,
        rel_error=rel_error,
    )
    return report, np.asarray(result)


def _run_solver_benchmarks(cases: Iterable[SolverMatrixCase]) -> list[SolverBenchmarkReport]:
    reports: list[SolverBenchmarkReport] = []
    specs = _solver_benchmark_specs()
    for case in cases:
        baseline: np.ndarray | None = None
        for spec in specs:
            report, result = _run_solver_once(case, spec, baseline)
            if spec.name == "scipy_fs" and result is not None:
                baseline = result
            reports.append(report)
    return reports


def _ratio_text(report: MatrixReport) -> str:
    if report.storage_bytes == 0:
        return "inf"
    ratio = report.dense_equiv_bytes / report.storage_bytes
    return f"{ratio:.2f}x"


def _write_table(reports: Iterable[MatrixReport], stream: TextIO) -> None:
    rows = list(reports)
    headers = ("name", "type", "shape", "dtype", "nnz", "storage", "dense equiv", "dense/storage", "density")
    rendered = []
    for report in rows:
        rendered.append(
            (
                report.name,
                report.kind,
                "x".join(str(axis) for axis in report.shape),
                report.dtype,
                "" if report.nnz is None else str(report.nnz),
                _format_bytes(report.storage_bytes),
                _format_bytes(report.dense_equiv_bytes),
                _ratio_text(report),
                "" if report.density is None else f"{report.density:.6g}",
            )
        )

    widths = [
        max(len(str(row[index])) for row in (headers, *rendered))
        for index in range(len(headers))
    ]
    stream.write("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + "\n")
    stream.write("  ".join("-" * width for width in widths) + "\n")
    for row in rendered:
        stream.write("  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row)) + "\n")


def _write_csv(reports: Iterable[MatrixReport], stream: TextIO) -> None:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "name",
            "type",
            "shape",
            "dtype",
            "nnz",
            "storage_bytes",
            "dense_equiv_bytes",
            "dense_to_storage_ratio",
            "density",
        ]
    )
    for report in reports:
        ratio = "" if report.storage_bytes == 0 else report.dense_equiv_bytes / report.storage_bytes
        writer.writerow(
            [
                report.name,
                report.kind,
                "x".join(str(axis) for axis in report.shape),
                report.dtype,
                "" if report.nnz is None else report.nnz,
                report.storage_bytes,
                report.dense_equiv_bytes,
                ratio,
                "" if report.density is None else report.density,
            ]
        )


def _number_cell(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _write_solver_csv(reports: Iterable[SolverBenchmarkReport], stream: TextIO) -> None:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "case",
            "backend",
            "status",
            "reason",
            "shape",
            "matrix_n",
            "matrix_nnz",
            "input_storage_bytes",
            "result_storage_bytes",
            "host_current_bytes",
            "host_peak_bytes",
            "gpu_free_before_bytes",
            "gpu_free_after_bytes",
            "gpu_memory_delta_bytes",
            "timing_build_s",
            "timing_analyze_s",
            "timing_factorize_s",
            "timing_solve_s",
            "timing_total_s",
            "max_abs_error",
            "rel_error",
        ]
    )
    for report in reports:
        writer.writerow(
            [
                report.case,
                report.backend,
                report.status,
                report.reason,
                report.shape,
                report.matrix_n,
                report.matrix_nnz,
                report.input_storage_bytes,
                _number_cell(report.result_storage_bytes),
                _number_cell(report.host_current_bytes),
                _number_cell(report.host_peak_bytes),
                _number_cell(report.gpu_free_before_bytes),
                _number_cell(report.gpu_free_after_bytes),
                _number_cell(report.gpu_memory_delta_bytes),
                _number_cell(report.timing_build_s),
                _number_cell(report.timing_analyze_s),
                _number_cell(report.timing_factorize_s),
                _number_cell(report.timing_solve_s),
                _number_cell(report.timing_total_s),
                _number_cell(report.max_abs_error),
                _number_cell(report.rel_error),
            ]
        )


def _write_solver_table(reports: Iterable[SolverBenchmarkReport], stream: TextIO) -> None:
    rows = list(reports)
    headers = (
        "case",
        "backend",
        "status",
        "time_s",
        "solve_s",
        "host_peak",
        "gpu_delta",
        "max_abs_err",
        "reason",
    )
    rendered = []
    for report in rows:
        rendered.append(
            (
                report.case,
                report.backend,
                report.status,
                _number_cell(report.timing_total_s),
                _number_cell(report.timing_solve_s),
                "" if report.host_peak_bytes is None else _format_bytes(report.host_peak_bytes),
                "" if report.gpu_memory_delta_bytes is None else _format_bytes(abs(report.gpu_memory_delta_bytes)),
                _number_cell(report.max_abs_error),
                report.reason,
            )
        )
    widths = [
        max(len(str(row[index])) for row in (headers, *rendered))
        for index in range(len(headers))
    ]
    stream.write("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + "\n")
    stream.write("  ".join("-" * width for width in widths) + "\n")
    for row in rendered:
        stream.write("  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row)) + "\n")


def _save_solver_outputs(reports: list[SolverBenchmarkReport], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mesti_solver_benchmark.csv"
    text_path = output_dir / "mesti_solver_benchmark.txt"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        _write_solver_csv(reports, stream)
    with text_path.open("w", encoding="utf-8") as stream:
        _write_solver_table(reports, stream)
    return csv_path, text_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ny", type=int, default=60, help="2D y grid size.")
    parser.add_argument("--nz", type=int, default=60, help="2D z grid size.")
    parser.add_argument("--nrhs", type=int, default=8, help="Number of B RHS columns.")
    parser.add_argument("--noutputs", type=int, default=None, help="Number of C projection rows. Defaults to --nrhs.")
    parser.add_argument("--surface-z", type=int, default=None, help="Source/projection z plane. Defaults to nz//2.")
    parser.add_argument("--wavelength", type=float, default=10.0, help="Wavelength used to compute k0dx.")
    parser.add_argument("--dx", type=float, default=1.0, help="Grid spacing used to compute k0dx.")
    parser.add_argument("--pml", type=int, default=0, help="PML pixels on each y/z side.")
    parser.add_argument("--ybc", default="PEC", help="Boundary condition along y.")
    parser.add_argument("--zbc", default="PEC", help="Boundary condition along z.")
    parser.add_argument("--no-upml", action="store_true", help="Disable UPML matrix assembly.")
    parser.add_argument("--format", choices=("table", "csv"), default="table", help="Output format.")
    parser.add_argument(
        "--solver-benchmark",
        action="store_true",
        help="Run solver timing/memory benchmarks instead of the storage-only matrix report.",
    )
    parser.add_argument(
        "--benchmark-small",
        default="small:8:8:2:2",
        help="Small benchmark case as name:ny:nz:nrhs:noutputs.",
    )
    parser.add_argument(
        "--benchmark-medium",
        default="medium:20:20:4:4",
        help="Medium benchmark case as name:ny:nz:nrhs:noutputs.",
    )
    parser.add_argument(
        "--benchmark-output-dir",
        type=Path,
        default=None,
        help="Optional directory for saved solver benchmark CSV and text reports.",
    )
    args = parser.parse_args()
    for name in ("ny", "nz", "nrhs"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive.")
    if args.noutputs is not None and args.noutputs <= 0:
        raise ValueError("--noutputs must be positive.")
    if args.pml < 0:
        raise ValueError("--pml must be non-negative.")
    return args


def main() -> None:
    args = _parse_args()
    if args.solver_benchmark:
        benchmark_cases = [
            _parse_solver_benchmark_case(args.benchmark_small),
            _parse_solver_benchmark_case(args.benchmark_medium),
        ]
        matrix_cases = [
            _build_solver_matrix_case(
                case,
                wavelength=args.wavelength,
                dx=args.dx,
                pml=args.pml,
                ybc=args.ybc,
                zbc=args.zbc,
                no_upml=args.no_upml,
            )
            for case in benchmark_cases
        ]
        reports = _run_solver_benchmarks(matrix_cases)
        if args.benchmark_output_dir is not None:
            _save_solver_outputs(reports, args.benchmark_output_dir)
        if args.format == "csv":
            _write_solver_csv(reports, sys.stdout)
        else:
            _write_solver_table(reports, sys.stdout)
        return

    reports = _build_reports(args)
    if args.format == "csv":
        _write_csv(reports, sys.stdout)
    else:
        _write_table(reports, sys.stdout)


if __name__ == "__main__":
    main()
