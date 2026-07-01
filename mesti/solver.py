"""Sparse linear solver wrapper for the Python MESTI port."""

from __future__ import annotations

import importlib.util
import os
import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from . import cudss_backend
from .types import Info, Matrices, Opts


_COMPLEX128_BYTES = np.dtype(np.complex128).itemsize
_DENSE_SOLVE_ARRAYS_PER_SPARSE_RHS_BATCH = 2
_SPARSE_RHS_BATCH_MEMORY_FRACTION = 0.05
_SPARSE_RHS_BATCH_MEMORY_CAP_BYTES = 512 * 1024**2
_SPARSE_RHS_BATCH_FALLBACK_BYTES = 256 * 1024**2


def _as_csc(matrix: Any, name: str) -> sparse.csc_matrix:
    if matrix is None:
        raise ValueError(f"matrices.{name} must not be None")
    if sparse.issparse(matrix):
        return matrix.astype(np.complex128, copy=False).tocsc()
    return sparse.csc_matrix(np.asarray(matrix, dtype=np.complex128))


def _as_rhs_matrix(matrix: Any, name: str) -> np.ndarray | sparse.csc_matrix:
    if matrix is None:
        raise ValueError(f"matrices.{name} must not be None")
    if sparse.issparse(matrix):
        return matrix.astype(np.complex128, copy=False).tocsc()
    rhs = np.asarray(matrix, dtype=np.complex128)
    if rhs.ndim == 1:
        rhs = rhs[:, np.newaxis]
    if rhs.ndim != 2:
        raise ValueError(f"matrices.{name} must be a vector or 2D matrix")
    return rhs


def _maybe_dense_or_sparse(matrix: Any, name: str) -> Any:
    if matrix is None:
        return None
    if sparse.issparse(matrix):
        return matrix.astype(np.complex128, copy=False).tocsc()
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError(f"matrices.{name} must be 2D when provided")
    return arr


def _is_transpose_b_projection(matrix: Any) -> bool:
    return isinstance(matrix, str) and matrix.replace(" ", "").lower() == "transpose(b)"


def _projection_from_transpose_b(B: np.ndarray | sparse.csc_matrix) -> np.ndarray | sparse.csc_matrix:
    if sparse.issparse(B):
        return B.transpose().tocsc()
    return np.asarray(B, dtype=np.complex128).T


def _true_scalar_option(value: Any, name: str) -> bool:
    if value is None:
        return False
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"opts.{name} must be a scalar")
    return bool(arr.reshape(-1)[0])


def _positive_int_option(value: Any, name: str) -> int:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"opts.{name} must be a positive integer scalar")
    scalar = arr.reshape(-1)[0]
    if np.iscomplexobj(arr) and not np.isclose(np.imag(scalar), 0):
        raise ValueError(f"opts.{name} must be a positive integer scalar")
    real_value = float(np.real(scalar))
    if real_value <= 0 or not real_value.is_integer():
        raise ValueError(f"opts.{name} must be a positive integer scalar")
    return int(real_value)


def _rhs_slice(
    matrix: np.ndarray | sparse.csc_matrix,
    start: int,
    stop: int,
    *,
    dense: bool,
) -> np.ndarray | sparse.csc_matrix:
    """Return one RHS batch, densifying only the selected sparse columns."""

    if sparse.issparse(matrix):
        batch = matrix[:, start:stop]
        if dense:
            return batch.toarray().astype(np.complex128, copy=False)
        return batch.astype(np.complex128, copy=False).tocsc()
    return np.asarray(matrix[:, start:stop], dtype=np.complex128)


def _available_memory_bytes() -> int | None:
    """Best-effort physical memory check using only the standard library."""

    if os.name == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            return None
        if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
            return int(pages * page_size)

    return None


def _sparse_rhs_batch_memory_budget_bytes() -> int:
    available = _available_memory_bytes()
    if available is None:
        return _SPARSE_RHS_BATCH_FALLBACK_BYTES
    budget = int(available * _SPARSE_RHS_BATCH_MEMORY_FRACTION)
    return max(1, min(_SPARSE_RHS_BATCH_MEMORY_CAP_BYTES, budget))


def _default_sparse_rhs_batch_size(B: sparse.csc_matrix, n_rhs: int) -> int:
    if n_rhs == 0:
        return 1

    # SciPy and mumpspy receive dense RHS batches, and the solve also produces
    # a dense X batch.  Budget for both so omitted nrhs improves throughput
    # without silently materializing a very wide sparse RHS at once.
    bytes_per_column = max(
        1,
        B.shape[0] * _COMPLEX128_BYTES * _DENSE_SOLVE_ARRAYS_PER_SPARSE_RHS_BATCH,
    )
    columns = _sparse_rhs_batch_memory_budget_bytes() // bytes_per_column
    return max(1, min(n_rhs, int(columns)))


def _default_batch_size(B: np.ndarray | sparse.csc_matrix, opts: Opts, n_rhs: int) -> int:
    if n_rhs == 0:
        return 1
    if opts.nrhs is not None:
        return min(_positive_int_option(opts.nrhs, "nrhs"), n_rhs)
    if sparse.issparse(B):
        return _default_sparse_rhs_batch_size(B, n_rhs)
    return n_rhs


def _mumps_binding_name() -> str | None:
    # Prefer mumpspy whenever both bindings are importable.  It is the binding
    # that passed the real-data WSL runs; python-mumps remains available as an
    # explicit backend for small tests and future debugging.
    if importlib.util.find_spec("mumpspy") is not None:
        return "mumpspy"
    if importlib.util.find_spec("mumps") is not None:
        return "python-mumps"
    return None


def _solver_backend(opts: Opts) -> str:
    requested = opts.solver.lower() if isinstance(opts.solver, str) else None
    if requested in {None, "", "auto"}:
        return "mumps" if _mumps_binding_name() is not None else "scipy"
    if requested in {"scipy", "splu", "superlu"}:
        return "scipy"
    if requested == "cudss":
        cudss_backend.require_available()
        return "cudss"
    if requested in {"mumps", "mumpspy", "python-mumps", "python_mumps"}:
        if requested == "mumpspy" and importlib.util.find_spec("mumpspy") is None:
            raise RuntimeError("opts.solver='mumpspy' requires `import mumpspy` in the active Python environment.")
        if requested in {"python-mumps", "python_mumps"} and importlib.util.find_spec("mumps") is None:
            raise RuntimeError(
                "opts.solver='python-mumps' requires `import mumps` from the python-mumps package "
                "in the active Python environment."
            )
        if _mumps_binding_name() is None:
            raise RuntimeError(
                "opts.solver='MUMPS' requires a Python MUMPS binding "
                "(`import mumps` from python-mumps or `import mumpspy`) "
                "in the active Python environment."
            )
        return "mumpspy" if requested == "mumpspy" else "python-mumps" if requested in {"python-mumps", "python_mumps"} else "mumps"
    raise ValueError("opts.solver must be 'auto', 'scipy', 'MUMPS', 'mumpspy', 'python-mumps', or 'cudss'.")


def _canonical_method(method: Any) -> str:
    if method is None:
        return "factorize_and_solve"
    if not isinstance(method, str):
        raise ValueError("opts.method must be a string when provided.")
    compact = method.replace(" ", "").lower()
    if compact in {"", "fs", "factorize_and_solve"}:
        return "factorize_and_solve"
    if compact == "apf":
        return "APF"
    if compact in {"fg", "c*inv(u)*inv(l)*b"}:
        return "C*inv(U)*inv(L)*B"
    raise ValueError("opts.method must be 'APF', 'FS'/'factorize_and_solve', or 'FG'/'C*inv(U)*inv(L)*B'.")


def _normalize_method(opts: Opts, c_present: bool) -> str:
    method = _canonical_method(opts.method)
    if method != "factorize_and_solve" and not c_present:
        raise ValueError("opts.method other than 'factorize_and_solve' requires matrices.C.")
    opts.method = method
    return method


def _backend_uses_python_mumps(backend: str) -> bool:
    return backend == "python-mumps" or (backend == "mumps" and _mumps_binding_name() == "python-mumps")


def _backend_uses_mumpspy(backend: str) -> bool:
    return backend == "mumpspy" or (backend == "mumps" and _mumps_binding_name() == "mumpspy")


def _has_cudss_controls(opts: Opts) -> bool:
    return (
        _true_scalar_option(opts.cudss_use_single_precision, "cudss_use_single_precision")
        or _true_scalar_option(opts.cudss_use_hybrid_memory, "cudss_use_hybrid_memory")
        or opts.cudss_hybrid_device_memory_limit is not None
        or opts.cudss_register_cuda_memory is not None
    )


def _validate_solver_options(opts: Opts, backend: str) -> None:
    if _true_scalar_option(opts.symmetrize_K, "symmetrize_K"):
        raise ValueError(
            "opts.symmetrize_K is not accepted by mesti_matrix_solver; use C='transpose(B)', "
            "opts.method='APF', a MUMPS/mumpspy solver, and a symmetric A instead."
        )
    if _true_scalar_option(opts.use_single_precision_MUMPS, "use_single_precision_MUMPS") and not _backend_uses_mumpspy(
        backend
    ):
        raise NotImplementedError(
            "opts.use_single_precision_MUMPS=true is supported only with the mumpspy backend."
        )
    if _has_cudss_controls(opts) and backend != "cudss":
        raise NotImplementedError("cuDSS optimization controls are supported only with opts.solver='cudss'.")
    if _true_scalar_option(opts.analysis_only, "analysis_only"):
        raise NotImplementedError("opts.analysis_only is not supported by the Python solver wrappers.")
    if _true_scalar_option(opts.store_ordering, "store_ordering"):
        raise NotImplementedError("opts.store_ordering is not supported by the Python solver wrappers.")
    if _true_scalar_option(opts.use_given_ordering, "use_given_ordering"):
        raise NotImplementedError("opts.use_given_ordering is not supported by the Python solver wrappers.")
    if opts.ordering is not None:
        raise NotImplementedError("opts.ordering is not supported by the Python solver wrappers.")
    if _true_scalar_option(opts.iterative_refinement, "iterative_refinement"):
        raise NotImplementedError("opts.iterative_refinement is not supported by the Python solver wrappers.")
    if _true_scalar_option(opts.use_BLR, "use_BLR") or opts.threshold_BLR is not None:
        raise NotImplementedError("MUMPS BLR controls are not supported by the Python solver wrappers.")
    if opts.icntl_36 is not None or opts.icntl_38 is not None:
        raise NotImplementedError("MUMPS BLR icntl_36/icntl_38 controls are not supported by the Python solver wrappers.")
    if opts.nthreads_OMP is not None:
        _positive_int_option(opts.nthreads_OMP, "nthreads_OMP")
        raise NotImplementedError("opts.nthreads_OMP is not supported by the Python solver wrappers.")
    if _true_scalar_option(opts.use_L0_threads, "use_L0_threads"):
        raise NotImplementedError("opts.use_L0_threads is not supported by the Python solver wrappers.")
    if _true_scalar_option(opts.use_METIS, "use_METIS") and not _backend_uses_python_mumps(backend):
        raise NotImplementedError("opts.use_METIS is only supported with the explicit python-mumps backend.")
    if _true_scalar_option(opts.write_LU_factor_to_disk, "write_LU_factor_to_disk") and not _backend_uses_python_mumps(
        backend
    ):
        raise NotImplementedError(
            "opts.write_LU_factor_to_disk is only supported with the explicit python-mumps backend."
        )


def _mumps_ordering(opts: Opts) -> str:
    if opts.use_METIS:
        return "metis"
    return "auto"


def _sparse_is_symmetric(A: sparse.csc_matrix) -> bool:
    difference = (A - A.transpose()).tocoo()
    if difference.nnz == 0:
        return True
    return bool(np.allclose(difference.data, 0.0, rtol=1e-13, atol=1e-13))


def _mumpspy_system_and_dtype(opts: Opts) -> tuple[str, Any]:
    if _true_scalar_option(opts.use_single_precision_MUMPS, "use_single_precision_MUMPS"):
        # mumpspy links the cmumps library as system="complex64"; keep the
        # public Python result cast to complex128 at the solver facade boundary.
        return "complex64", np.complex64
    return "complex", np.complex128


def _as_apf_block(matrix: np.ndarray | sparse.csc_matrix, name: str, dtype: Any = np.complex128) -> sparse.csc_matrix:
    if sparse.issparse(matrix):
        return matrix.astype(dtype, copy=False).tocsc()
    arr = np.asarray(matrix, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"matrices.{name} must be 2D for APF.")
    return sparse.csc_matrix(arr)


def _matrix_nnz(matrix: np.ndarray | sparse.csc_matrix) -> int:
    if sparse.issparse(matrix):
        return int(matrix.nnz)
    return int(np.count_nonzero(matrix))


def _solve_with_scipy(
    A: sparse.csc_matrix,
    B: np.ndarray | sparse.csc_matrix,
    C: np.ndarray | sparse.csc_matrix | None,
    opts: Opts,
    result_shape: tuple[int, int],
    info: Info,
) -> np.ndarray:
    start_solve = time.perf_counter()
    lu = spla.splu(A)
    batch_size = _default_batch_size(B, opts, result_shape[1])
    X = np.empty(result_shape, dtype=np.complex128)
    for batch_start in range(0, result_shape[1], batch_size):
        batch_stop = min(batch_start + batch_size, result_shape[1])
        X_batch = lu.solve(_rhs_slice(B, batch_start, batch_stop, dense=True))
        if C is not None:
            X_batch = C @ X_batch
        X[:, batch_start:batch_stop] = np.asarray(X_batch, dtype=np.complex128)
    info.timing_solve = time.perf_counter() - start_solve
    return X


def _solve_with_scipy_fg(
    A: sparse.csc_matrix,
    B: np.ndarray | sparse.csc_matrix,
    C: np.ndarray | sparse.csc_matrix,
    opts: Opts,
    result_shape: tuple[int, int],
    info: Info,
) -> np.ndarray:
    start_solve = time.perf_counter()
    lu = spla.splu(A)
    X = np.empty(result_shape, dtype=np.complex128)

    if _matrix_nnz(C) < _matrix_nnz(B):
        C_rhs = C.transpose().toarray() if sparse.issparse(C) else np.asarray(C, dtype=np.complex128).T
        C_inv_A = lu.solve(C_rhs, trans="T").T
        if sparse.issparse(B):
            X[:, :] = (B.transpose() @ C_inv_A.T).T
        else:
            X[:, :] = C_inv_A @ np.asarray(B, dtype=np.complex128)
    else:
        batch_size = _default_batch_size(B, opts, result_shape[1])
        for batch_start in range(0, result_shape[1], batch_size):
            batch_stop = min(batch_start + batch_size, result_shape[1])
            X_batch = lu.solve(_rhs_slice(B, batch_start, batch_stop, dense=True))
            X[:, batch_start:batch_stop] = C @ X_batch

    info.timing_solve = time.perf_counter() - start_solve
    return X


def _solve_with_python_mumps(
    A: sparse.csc_matrix,
    B: np.ndarray | sparse.csc_matrix,
    C: np.ndarray | sparse.csc_matrix | None,
    opts: Opts,
    result_shape: tuple[int, int],
    info: Info,
) -> np.ndarray:
    import mumps

    verbose = bool(opts.verbal_solver if opts.verbal_solver is not None else opts.verbal)
    symmetric = bool(opts.is_symmetric_A) if opts.is_symmetric_A is not None else False
    ooc = bool(opts.write_LU_factor_to_disk) if opts.write_LU_factor_to_disk is not None else False

    with mumps.Context(verbose=verbose) as ctx:
        ctx.set_matrix(A, symmetric=symmetric)

        start_analyze = time.perf_counter()
        ctx.analyze(ordering=_mumps_ordering(opts))
        info.timing_analyze = time.perf_counter() - start_analyze

        start_factor = time.perf_counter()
        ctx.factor(reuse_analysis=True, ooc=ooc)
        info.timing_factorize = time.perf_counter() - start_factor

        start_solve = time.perf_counter()
        batch_size = _default_batch_size(B, opts, result_shape[1])
        X = np.empty(result_shape, dtype=np.complex128)
        for batch_start in range(0, result_shape[1], batch_size):
            batch_stop = min(batch_start + batch_size, result_shape[1])
            X_batch = ctx.solve(_rhs_slice(B, batch_start, batch_stop, dense=False))
            if C is not None:
                X_batch = C @ X_batch
            X[:, batch_start:batch_stop] = np.asarray(X_batch, dtype=np.complex128)
        info.timing_solve = time.perf_counter() - start_solve

    return X


def _solve_with_mumpspy(
    A: sparse.csc_matrix,
    B: np.ndarray | sparse.csc_matrix,
    C: np.ndarray | sparse.csc_matrix | None,
    opts: Opts,
    result_shape: tuple[int, int],
    info: Info,
) -> np.ndarray:
    from mumpspy import MumpsSolver

    verbose = bool(opts.verbal_solver if opts.verbal_solver is not None else opts.verbal)
    symmetric = bool(opts.is_symmetric_A) if opts.is_symmetric_A is not None else False
    system, mumps_dtype = _mumpspy_system_and_dtype(opts)
    A_mumps = A.astype(mumps_dtype, copy=False)

    solver = MumpsSolver(is_sym=symmetric, system=system, silent=not verbose)
    try:
        start_factor = time.perf_counter()
        solver.set_mtx(A_mumps)
        info.timing_factorize = time.perf_counter() - start_factor

        start_solve = time.perf_counter()
        batch_size = _default_batch_size(B, opts, result_shape[1])
        X = np.empty(result_shape, dtype=np.complex128)
        for batch_start in range(0, result_shape[1], batch_size):
            batch_stop = min(batch_start + batch_size, result_shape[1])
            rhs = _rhs_slice(B, batch_start, batch_stop, dense=True).astype(mumps_dtype, copy=False)
            X_batch = solver.solve(rhs)
            if C is not None:
                X_batch = C @ X_batch
            X[:, batch_start:batch_stop] = np.asarray(X_batch, dtype=np.complex128)
        info.timing_solve = time.perf_counter() - start_solve
    finally:
        solver.__del__()

    return X


def _solve_with_cudss(
    A: sparse.csc_matrix,
    B: np.ndarray | sparse.csc_matrix,
    C: np.ndarray | sparse.csc_matrix | None,
    opts: Opts,
    result_shape: tuple[int, int],
    info: Info,
) -> np.ndarray:
    batch_size = _default_batch_size(B, opts, result_shape[1])
    X = np.empty(result_shape, dtype=np.complex128)
    timing_analyze = 0.0
    timing_factorize = 0.0
    timing_solve = 0.0
    saw_analyze = False
    saw_factorize = False
    saw_solve = False

    for batch_start in range(0, result_shape[1], batch_size):
        batch_stop = min(batch_start + batch_size, result_shape[1])
        batch_info = Info(opts=opts)
        X_batch = cudss_backend.cudss_solve(A, _rhs_slice(B, batch_start, batch_stop, dense=False), opts, batch_info)
        if C is not None:
            X_batch = C @ X_batch
        X[:, batch_start:batch_stop] = np.asarray(X_batch, dtype=np.complex128)

        if batch_info.timing_analyze is not None:
            timing_analyze += batch_info.timing_analyze
            saw_analyze = True
        if batch_info.timing_factorize is not None:
            timing_factorize += batch_info.timing_factorize
            saw_factorize = True
        if batch_info.timing_solve is not None:
            timing_solve += batch_info.timing_solve
            saw_solve = True

    if saw_analyze:
        info.timing_analyze = timing_analyze
    if saw_factorize:
        info.timing_factorize = timing_factorize
    if saw_solve:
        info.timing_solve = timing_solve
    return X


def _solve_with_mumpspy_apf(
    A: sparse.csc_matrix,
    B: np.ndarray | sparse.csc_matrix,
    C: np.ndarray | sparse.csc_matrix,
    opts: Opts,
    transpose_B: bool,
    info: Info,
) -> np.ndarray:
    from mumpspy import MumpsSolver

    verbose = bool(opts.verbal_solver if opts.verbal_solver is not None else opts.verbal)
    system, mumps_dtype = _mumpspy_system_and_dtype(opts)
    A_mumps = A.astype(mumps_dtype, copy=False)
    B_block = _as_apf_block(B, "B", mumps_dtype)
    C_block = B_block.transpose().tocsc() if transpose_B else _as_apf_block(C, "C", mumps_dtype)

    n = A.shape[0]
    n_in = B_block.shape[1]
    n_out = C_block.shape[0]
    n_schur = max(n_in, n_out)
    if n_in < n_schur:
        B_block = sparse.hstack(
            [B_block, sparse.csc_matrix((n, n_schur - n_in), dtype=mumps_dtype)],
            format="csc",
        )
    if n_out < n_schur:
        C_block = sparse.vstack(
            [C_block, sparse.csc_matrix((n_schur - n_out, n), dtype=mumps_dtype)],
            format="csc",
        )

    start_build = time.perf_counter()
    zero = sparse.csc_matrix((n_schur, n_schur), dtype=mumps_dtype)
    K = sparse.bmat([[A_mumps, B_block], [C_block, zero]], format="csc")
    info.timing_build = time.perf_counter() - start_build

    if transpose_B:
        symmetric = bool(opts.is_symmetric_A) if opts.is_symmetric_A is not None else _sparse_is_symmetric(A)
    else:
        symmetric = False

    solver = MumpsSolver(is_sym=symmetric, system=system, silent=not verbose)
    try:
        start_factor = time.perf_counter()
        solver.set_mtx(K, factorize=False)
        # MUMPS returns the Schur complement of K, i.e. -C*inv(A)*B for the
        # zero lower-right block.  Julia's APF path negates the Schur block too.
        schur_indices = np.arange(n + 1, n + n_schur + 1, dtype=np.int32)
        schur = solver.schur_complement(schur_indices)
        info.timing_factorize = time.perf_counter() - start_factor
        info.timing_solve = 0.0
    finally:
        solver.__del__()

    # mumpspy reshapes MUMPS' Fortran-ordered Schur buffer as a C-ordered
    # array, so transpose once at the binding boundary to recover Julia's
    # row/column convention.
    S = -np.asarray(schur, dtype=np.complex128).T
    return S[:n_out, :n_in]


def _solve_with_apf(
    A: sparse.csc_matrix,
    B: np.ndarray | sparse.csc_matrix,
    C: np.ndarray | sparse.csc_matrix | None,
    opts: Opts,
    transpose_B: bool,
    backend: str,
    info: Info,
) -> np.ndarray:
    binding = _mumps_binding_name() if backend == "mumps" else backend
    if binding == "cudss":
        if C is None and not transpose_B:
            raise ValueError("opts.method='APF' requires matrices.C.")
        return cudss_backend.cudss_apf(A, B, C, opts, transpose_B, info)
    if binding == "mumpspy":
        if C is None and not transpose_B:
            raise ValueError("opts.method='APF' requires matrices.C.")
        return _solve_with_mumpspy_apf(A, B, C, opts, transpose_B, info)
    if binding in {"python-mumps", "python_mumps"}:
        raise NotImplementedError("opts.method='APF' requires the mumpspy binding; python-mumps APF is not supported.")
    raise RuntimeError("opts.method='APF' requires opts.solver='MUMPS' or 'mumpspy' with mumpspy installed.")


def _solve_with_mumps(
    A: sparse.csc_matrix,
    B: np.ndarray | sparse.csc_matrix,
    C: np.ndarray | sparse.csc_matrix | None,
    opts: Opts,
    result_shape: tuple[int, int],
    info: Info,
) -> np.ndarray:
    binding = _mumps_binding_name()
    if binding == "python-mumps":
        return _solve_with_python_mumps(A, B, C, opts, result_shape, info)
    if binding == "mumpspy":
        return _solve_with_mumpspy(A, B, C, opts, result_shape, info)
    raise RuntimeError(
        "opts.solver='MUMPS' requires a Python MUMPS binding "
        "(`import mumps` from python-mumps or `import mumpspy`) "
        "in the active Python environment."
    )


def mesti_matrix_solver(
    matrices: Matrices,
    opts: Opts | None = None,
) -> tuple[np.ndarray, Info]:
    """Solve the matrix systems used by MESTI.

    For the initial Python port, this implements the direct SciPy equivalent of
    the Julia fallback path:

    - with ``A`` and ``B`` only, return ``X = inv(A) * B``;
    - with ``C`` provided, return ``C * inv(A) * B``;
    - with ``D`` also provided, return ``C * inv(A) * B - D``.

    Sparse RHS matrices are sliced before any dense conversion, so projected
    solves never materialize the full ``B`` matrix just to pass a batch into a
    direct solver.  ``opts.nrhs`` controls the batch width when provided.  When
    it is omitted, dense RHS inputs use all columns and sparse RHS inputs use a
    conservative memory-aware batch heuristic based on the dense RHS slice plus
    dense solution slice that SciPy and ``mumpspy`` need during each solve.

    ``opts.solver`` may be ``"auto"``/``None``, ``"scipy"``, ``"MUMPS"``,
    ``"mumpspy"``, ``"python-mumps"``, or explicit ``"cudss"``.
    Auto mode uses a Python MUMPS binding when one is available in the active
    Python environment.  The preferred binding is ``mumpspy`` because it passed
    the larger WSL real-data runs in this project; ``python-mumps``
    (``import mumps``) is also supported but should be requested explicitly
    when debugging that backend.  If neither binding is importable, auto mode
    falls back to SciPy/SuperLU.  The default Python wrappers solve complex
    systems in double precision; explicit
    ``opts.use_single_precision_MUMPS = true`` is supported only for the
    ``mumpspy`` backend, where it selects the complex64 MUMPS library.  Compare
    against Julia references generated with
    ``opts.use_single_precision_MUMPS = false`` when checking double-precision
    real-data parity at tight tolerances.

    ``opts.method="FG"`` / ``"C*inv(U)*inv(L)*B"`` is available for explicit
    SciPy/SuperLU projected solves.  ``opts.method="APF"`` is available only
    through backend implementations that expose a Schur-complement path
    (currently ``mumpspy``; cuDSS dispatch is scaffolded and its numerical
    implementation is deferred).

    Julia MESTI exposes raw ``Mumps`` objects and MPI-oriented helper
    functions.  The Python port intentionally keeps that layer delegated to
    external Python bindings and exposes only this solver facade.
    """

    if not isinstance(matrices, Matrices):
        raise TypeError("matrices must be a Matrices instance")

    opts = opts if opts is not None else Opts()
    info = Info(opts=opts)
    start_total = time.perf_counter()
    transpose_B = _is_transpose_b_projection(matrices.C)
    if isinstance(matrices.C, str) and not transpose_B:
        raise ValueError('matrices.C must be numeric, sparse, or "transpose(B)".')
    method = _normalize_method(opts, matrices.C is not None)

    A = _as_csc(matrices.A, "A")
    B = _as_rhs_matrix(matrices.B, "B")
    C = _projection_from_transpose_b(B) if transpose_B else _maybe_dense_or_sparse(matrices.C, "C")
    D = _maybe_dense_or_sparse(matrices.D, "D")

    if A.shape[0] != A.shape[1]:
        raise ValueError("matrices.A must be square")
    if A.shape[0] != B.shape[0]:
        raise ValueError("matrices.A row count must match matrices.B row count")
    if C is not None and C.shape[1] != A.shape[0]:
        raise ValueError("matrices.C column count must match solution row count")

    n_rhs = B.shape[1]
    result_rows = C.shape[0] if C is not None else A.shape[0]
    result_shape = (result_rows, n_rhs)
    if D is not None and D.shape != result_shape:
        raise ValueError("matrices.D shape must match projected solution shape")

    backend = _solver_backend(opts)
    _validate_solver_options(opts, backend)
    opts.solver = backend
    if method == "APF":
        X = _solve_with_apf(A, B, C, opts, transpose_B, backend, info)
    elif method == "C*inv(U)*inv(L)*B":
        if backend != "scipy":
            raise NotImplementedError(
                "opts.method='FG' / 'C*inv(U)*inv(L)*B' is implemented only for opts.solver='scipy'."
            )
        X = _solve_with_scipy_fg(A, B, C, opts, result_shape, info)
    elif backend == "mumps":
        X = _solve_with_mumps(A, B, C, opts, result_shape, info)
    elif backend == "mumpspy":
        X = _solve_with_mumpspy(A, B, C, opts, result_shape, info)
    elif backend == "python-mumps":
        X = _solve_with_python_mumps(A, B, C, opts, result_shape, info)
    elif backend == "cudss":
        X = _solve_with_cudss(A, B, C, opts, result_shape, info)
    else:
        X = _solve_with_scipy(A, B, C, opts, result_shape, info)

    if D is not None:
        D_arr = D.toarray() if sparse.issparse(D) else D
        X = X - D_arr

    if X.shape[1] == 1:
        X = X.reshape(-1, 1)

    info.timing_total = time.perf_counter() - start_total
    return np.asarray(X, dtype=np.complex128), info
