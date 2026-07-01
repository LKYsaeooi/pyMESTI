"""Compatibility facade for Julia MESTI's raw MUMPS helper API.

The Julia package exposes a low-level ``Mumps`` object plus many mutating
helpers whose names end in ``!``.  Python cannot expose those identifiers
directly, so this module uses the same names without the bang suffix.  The
stateful C/MPI invocation surface remains explicit unsupported behavior, while
the high-level algebra helpers use SciPy/SuperLU so small compatibility tests
and translated examples can call the expected public names.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


DEFAULT_FORTRAN_COMMUNICATOR = -987654
ICNTL_DEFAULT = (
    6,
    0,
    6,
    2,
    0,
    7,
    7,
    77,
    1,
    0,
    0,
    1,
    0,
    20,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    -32,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    333,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
)


class UnsupportedMumpsOperation(NotImplementedError):
    """Raised when Julia's raw MUMPS C/MPI behavior is intentionally absent."""


class Mumps:
    """Small Python state object mirroring Julia's public ``Mumps`` wrapper.

    ``Mumps(A, rhs=None, sym=None, par=1)`` stores a square matrix and optional
    RHS.  Solves and factorization use SciPy/SuperLU; raw MUMPS C invocation,
    pointer loading, distributed matrices, and MPI worker orchestration are not
    reimplemented.
    """

    def __init__(
        self,
        A: Any = None,
        rhs: Any = None,
        *,
        sym: int | None = None,
        par: int = 1,
        comm: int = DEFAULT_FORTRAN_COMMUNICATOR,
    ) -> None:
        self.sym = 0 if sym is None else int(sym)
        self.par = int(par)
        self.comm = int(comm)
        self.job = 0
        self.icntl = list(ICNTL_DEFAULT)
        self.cntl = [0.0] * 15
        self.keep = [0] * 500
        self.save_dir = ""
        self.save_prefix = ""
        self.A: sparse.csc_matrix | None = None
        self.rhs: np.ndarray | sparse.csc_matrix | None = None
        self.solution: np.ndarray | None = None
        self.factor: spla.SuperLU | None = None
        self.schur: np.ndarray | None = None
        self._finalized = False
        self.n = 0
        self.nnz = 0
        self.nrhs = 0
        self.lrhs = 0
        self.nz_rhs = 0
        self.size_schur = 0
        if A is not None:
            if sym is None:
                matrix = _as_csc(A)
                self.sym = 2 if _sparse_is_symmetric(matrix) else 0
            provide_matrix(self, A)
        if rhs is not None:
            provide_rhs(self, rhs)

    def __repr__(self) -> str:
        state = "finalized" if self._finalized else f"job={self.job}"
        return f"Mumps(n={self.n}, nnz={self.nnz}, nrhs={self.nrhs}, sym={self.sym}, {state})"


def _as_csc(matrix: Any) -> sparse.csc_matrix:
    if sparse.issparse(matrix):
        result = matrix.astype(np.complex128, copy=False).tocsc()
    else:
        result = sparse.csc_matrix(np.asarray(matrix, dtype=np.complex128))
    if result.shape[0] != result.shape[1]:
        raise ValueError("matrix must be square")
    return result


def _rhs_matrix(rhs: Any) -> np.ndarray | sparse.csc_matrix:
    if sparse.issparse(rhs):
        result = rhs.astype(np.complex128, copy=False).tocsc()
        if result.ndim != 2:
            raise ValueError("rhs must be a vector or 2D matrix")
        return result
    result = np.asarray(rhs, dtype=np.complex128)
    if result.ndim == 1:
        result = result[:, np.newaxis]
    if result.ndim != 2:
        raise ValueError("rhs must be a vector or 2D matrix")
    return result


def _rhs_dense(rhs: np.ndarray | sparse.csc_matrix) -> np.ndarray:
    return rhs.toarray().astype(np.complex128, copy=False) if sparse.issparse(rhs) else np.asarray(rhs)


def _check_finalized(mumps: Mumps) -> None:
    if mumps._finalized:
        raise RuntimeError("Mumps object already finalized")


def _require_matrix(mumps: Mumps) -> sparse.csc_matrix:
    _check_finalized(mumps)
    if mumps.A is None:
        raise ValueError("matrix not yet provided to mumps object")
    return mumps.A


def _require_rhs(mumps: Mumps) -> np.ndarray | sparse.csc_matrix:
    _check_finalized(mumps)
    if mumps.rhs is None:
        raise ValueError("rhs not yet provided to mumps object")
    return mumps.rhs


def _set_1_based(values: list[Any], i: int, value: Any, name: str) -> None:
    index = int(i) - 1
    if index < 0 or index >= len(values):
        raise ValueError(f"{name} index {i} is outside 1:{len(values)}")
    values[index] = value


def _sparse_is_symmetric(A: sparse.csc_matrix) -> bool:
    difference = (A - A.transpose()).tocoo()
    if difference.nnz == 0:
        return True
    return bool(np.allclose(difference.data, 0.0, rtol=1e-13, atol=1e-13))


def _indices_from_selector(selector: Any, n: int) -> np.ndarray:
    if sparse.issparse(selector):
        indices = np.unique(selector.tocoo().row)
    else:
        indices = np.asarray(selector, dtype=int).reshape(-1)
    if indices.size == 0:
        raise ValueError("Schur/selected-inverse indices must be nonempty")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("indices are zero-based and must be within the matrix shape")
    return np.unique(indices)


def invoke_mumps_unsafe(mumps: Mumps) -> None:
    raise UnsupportedMumpsOperation(
        "Raw MUMPS C/MPI invocation is not exposed by the Python port; use "
        "mesti_matrix_solver, mumps_solve, or an external Python MUMPS binding."
    )


def invoke_mumps(mumps: Mumps) -> None:
    _check_finalized(mumps)
    invoke_mumps_unsafe(mumps)


def set_keep(mumps: Mumps, i: int, val: int, *, displaylevel: int = 0) -> None:
    _check_finalized(mumps)
    _set_1_based(mumps.keep, i, int(val), "KEEP")


def set_icntl(mumps: Mumps, i: int, val: int, *, displaylevel: int = 0) -> None:
    _check_finalized(mumps)
    _set_1_based(mumps.icntl, i, int(val), "ICNTL")


def set_cntl(mumps: Mumps, i: int, val: float, *, displaylevel: int = 0) -> None:
    _check_finalized(mumps)
    _set_1_based(mumps.cntl, i, float(val), "CNTL")


def set_job(mumps: Mumps, i: int) -> None:
    _check_finalized(mumps)
    mumps.job = int(i)


def set_save_dir(mumps: Mumps, directory: str) -> None:
    _check_finalized(mumps)
    if len(directory) > 255:
        raise ValueError("directory name must contain at most 255 characters")
    mumps.save_dir = str(directory)


def set_save_prefix(mumps: Mumps, prefix: str) -> None:
    _check_finalized(mumps)
    if len(prefix) > 255:
        raise ValueError("prefix name must contain at most 255 characters")
    mumps.save_prefix = str(prefix)


def provide_matrix(mumps: Mumps, A: Any) -> None:
    _check_finalized(mumps)
    matrix = _as_csc(A)
    mumps.A = matrix
    mumps.factor = None
    mumps.schur = None
    mumps.solution = None
    mumps.n = matrix.shape[0]
    mumps.nnz = int(matrix.nnz)


def provide_rhs(mumps: Mumps, rhs: Any) -> None:
    _check_finalized(mumps)
    result = _rhs_matrix(rhs)
    if mumps.A is not None and result.shape[0] != mumps.A.shape[0]:
        raise ValueError("rhs row count must match matrix size")
    mumps.rhs = result
    mumps.solution = None
    mumps.lrhs = result.shape[0]
    mumps.nrhs = result.shape[1]
    mumps.nz_rhs = int(result.nnz) if sparse.issparse(result) else int(np.count_nonzero(result))


def provide_rhs_sparse(mumps: Mumps, rhs: Any) -> None:
    provide_rhs(mumps, sparse.csc_matrix(rhs, dtype=np.complex128))


def provide_rhs_dense(mumps: Mumps, rhs: Any) -> None:
    provide_rhs(mumps, np.asarray(rhs, dtype=np.complex128))


def get_rhs(mumps: Mumps) -> np.ndarray | sparse.csc_matrix:
    rhs = _require_rhs(mumps)
    return rhs.copy()


def get_rhs_into(out: Any, mumps: Mumps) -> None:
    rhs = get_rhs(mumps)
    if sparse.issparse(out):
        copied = rhs.tocsc() if sparse.issparse(rhs) else sparse.csc_matrix(rhs)
        out.data[:] = copied.data
        out.indices[:] = copied.indices
        out.indptr[:] = copied.indptr
    else:
        np.asarray(out)[:] = _rhs_dense(rhs)


def get_sol(mumps: Mumps) -> np.ndarray:
    _check_finalized(mumps)
    if mumps.solution is None:
        raise ValueError("mumps has not passed through a solution phase")
    return np.asarray(mumps.solution, dtype=np.complex128).copy()


def get_sol_into(out: Any, mumps: Mumps) -> None:
    np.asarray(out)[:] = get_sol(mumps)


def _ensure_factorized(mumps: Mumps) -> spla.SuperLU:
    A = _require_matrix(mumps)
    if mumps.factor is None:
        mumps.factor = spla.splu(A)
        mumps.job = 4
    return mumps.factor


def mumps_factorize_inplace(mumps: Mumps) -> None:
    _ensure_factorized(mumps)
    mumps.job = 4


def mumps_factorize(A: Any) -> Mumps:
    mumps = A if isinstance(A, Mumps) else Mumps(A)
    mumps_factorize_inplace(mumps)
    return mumps


def mumps_solve_inplace(*args: Any, **kwargs: Any) -> None:
    if len(args) < 2:
        raise TypeError("mumps_solve_inplace expects an output array plus solve arguments")
    out = args[0]
    np.asarray(out)[:] = mumps_solve(*args[1:], **kwargs)


def mumps_solve(*args: Any, **kwargs: Any) -> np.ndarray:
    """Solve ``A x = rhs`` using a Julia-compatible call surface.

    Accepted forms are ``mumps_solve(A, rhs)``, ``mumps_solve(mumps)``, and
    ``mumps_solve(mumps, rhs)``.
    """

    if not args:
        raise TypeError("mumps_solve expects a Mumps object or matrix and rhs")
    first = args[0]
    if isinstance(first, Mumps):
        mumps = first
        if len(args) > 2:
            raise TypeError("mumps_solve(mumps, rhs) accepts at most two positional arguments")
        if len(args) == 2:
            provide_rhs(mumps, args[1])
        elif mumps.solution is not None:
            return get_sol(mumps)
        rhs = _require_rhs(mumps)
    else:
        if len(args) != 2:
            raise TypeError("mumps_solve(A, rhs) requires exactly two positional arguments")
        mumps = Mumps(first, args[1], **kwargs)
        rhs = _require_rhs(mumps)

    factor = _ensure_factorized(mumps)
    solution = factor.solve(_rhs_dense(rhs))
    mumps.solution = np.asarray(solution, dtype=np.complex128)
    mumps.job = 3
    return mumps.solution.copy()


def mumps_det_inplace(mumps: Mumps, *, discard: bool | int = True) -> None:
    mumps_factorize_inplace(mumps)
    set_icntl(mumps, 33, 1)


def mumps_det(A: Any) -> complex:
    mumps = A if isinstance(A, Mumps) else Mumps(A)
    matrix = _require_matrix(mumps)
    return complex(np.linalg.det(matrix.toarray()))


def set_schur_centralized_by_column(mumps: Mumps, schur_inds: Any) -> None:
    matrix = _require_matrix(mumps)
    indices = _indices_from_selector(schur_inds, matrix.shape[0])
    mumps.size_schur = int(indices.size)
    mumps._schur_indices = indices


def mumps_schur_complement_inplace(mumps: Mumps, schur_inds: Any) -> None:
    A = _require_matrix(mumps).toarray()
    indices = _indices_from_selector(schur_inds, A.shape[0])
    rest = np.setdiff1d(np.arange(A.shape[0]), indices, assume_unique=True)
    if rest.size == 0:
        schur = A[np.ix_(indices, indices)]
    else:
        A_ii = A[np.ix_(indices, indices)]
        A_ir = A[np.ix_(indices, rest)]
        A_ri = A[np.ix_(rest, indices)]
        A_rr = A[np.ix_(rest, rest)]
        schur = A_ii - A_ir @ np.linalg.solve(A_rr, A_ri)
    mumps.schur = np.asarray(schur, dtype=np.complex128)
    mumps.size_schur = int(indices.size)
    mumps.job = 4


def get_schur_complement(mumps: Mumps) -> np.ndarray:
    _check_finalized(mumps)
    if mumps.schur is None:
        raise ValueError("schur complement not yet allocated")
    return mumps.schur.copy()


def get_schur_complement_into(out: Any, mumps: Mumps) -> None:
    np.asarray(out)[:] = get_schur_complement(mumps)


def mumps_schur_complement(A: Any, x: Any) -> np.ndarray:
    mumps = A if isinstance(A, Mumps) else Mumps(A)
    mumps_schur_complement_inplace(mumps, x)
    return get_schur_complement(mumps)


def mumps_select_inv_inplace(x: sparse.spmatrix, A: Any) -> None:
    if not sparse.issparse(x):
        raise TypeError("mumps_select_inv_inplace target must be sparse")
    result = mumps_select_inv(A, x).tocsc()
    target = x.tocsc()
    if target.shape != result.shape or target.nnz != result.nnz:
        raise ValueError("target sparsity pattern changed during selected inverse")
    target.data[:] = result.data
    if hasattr(x, "data"):
        x.data[:] = target.data


def mumps_select_inv(A: Any, x: Any, J: Any = None) -> sparse.csc_matrix:
    mumps = A if isinstance(A, Mumps) else Mumps(A)
    matrix = _require_matrix(mumps)
    inv_a = np.linalg.inv(matrix.toarray())
    if J is None and sparse.issparse(x):
        coo = x.tocoo()
        rows = coo.row
        cols = coo.col
    elif J is not None:
        rows = np.asarray(x, dtype=int).reshape(-1)
        cols = np.asarray(J, dtype=int).reshape(-1)
        if rows.shape != cols.shape:
            raise ValueError("I and J must have the same length")
    else:
        raise TypeError("mumps_select_inv expects a sparse pattern or I, J index arrays")
    if np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= matrix.shape[0]) or np.any(cols >= matrix.shape[1]):
        raise ValueError("indices are zero-based and must be within the matrix shape")
    values = inv_a[rows, cols]
    return sparse.coo_matrix((values, (rows, cols)), shape=matrix.shape).tocsc()


def initialize(mumps: Mumps) -> None:
    mumps._finalized = False
    mumps.job = -1


def finalize(mumps: Mumps) -> None:
    _check_finalized(mumps)
    mumps._finalized = True
    mumps.job = -2
    mumps.factor = None


def finalize_unsafe(mumps: Mumps) -> None:
    mumps._finalized = True
    mumps.job = -2
    mumps.factor = None


def default_icntl(mumps: Mumps) -> None:
    _check_finalized(mumps)
    mumps.icntl = list(ICNTL_DEFAULT)


def set_error_stream(mumps: Mumps, i: int) -> None:
    set_icntl(mumps, 1, i)


def set_diagnostics_stream(mumps: Mumps, i: int) -> None:
    set_icntl(mumps, 2, i)


def set_info_stream(mumps: Mumps, i: int) -> None:
    set_icntl(mumps, 3, i)


def set_print_level(mumps: Mumps, i: int) -> None:
    set_icntl(mumps, 4, i)


def suppress_printing(mumps: Mumps) -> None:
    set_print_level(mumps, 1)


def toggle_printing(mumps: Mumps) -> None:
    set_print_level(mumps, 2 if mumps.icntl[3] == 1 else 1)


suppress_display = suppress_printing
toggle_display = toggle_printing


def sparse_matrix(mumps: Mumps) -> None:
    set_icntl(mumps, 5, 0)


def dense_matrix(mumps: Mumps) -> None:
    set_icntl(mumps, 5, 1)


def transpose_mumps(mumps: Mumps) -> None:
    set_icntl(mumps, 9, 1 if mumps.icntl[8] == 0 else 0)


def sparse_rhs(mumps: Mumps) -> None:
    set_icntl(mumps, 20, 1)


def dense_rhs(mumps: Mumps) -> None:
    set_icntl(mumps, 20, 0)


def toggle_null_pivot(mumps: Mumps) -> None:
    set_icntl(mumps, 24, 1 if mumps.icntl[23] == 0 else 0)


def is_matrix_assembled(mumps: Mumps) -> bool:
    return mumps.icntl[4] not in {1}


def is_matrix_distributed(mumps: Mumps) -> bool:
    return mumps.icntl[17] in {1, 2, 3}


def is_rhs_dense(mumps: Mumps) -> bool:
    return mumps.icntl[19] not in {1, 2, 3}


def is_sol_central(mumps: Mumps) -> bool:
    return mumps.icntl[20] not in {1}


def has_det(mumps: Mumps) -> bool:
    return mumps.icntl[32] != 0


def is_symmetric(mumps: Mumps) -> bool:
    return mumps.sym in {1, 2}


def is_posdef(mumps: Mumps) -> bool:
    return mumps.sym == 1


def has_matrix(mumps: Mumps) -> bool:
    return mumps.A is not None and mumps.n > 0


def has_rhs(mumps: Mumps) -> bool:
    return mumps.rhs is not None and (mumps.nrhs * mumps.lrhs > 0 or mumps.nz_rhs > 0)


def has_schur(mumps: Mumps) -> bool:
    return mumps.schur is not None and mumps.size_schur > 0


def _display_values(name: str, values: list[Any] | tuple[Any, ...]) -> str:
    lines = [f"{name} settings:"]
    for index, value in enumerate(values, start=1):
        lines.append(f"{index},\t{value}")
    return "\n".join(lines)


def display_icntl(mumps_or_values: Mumps | list[Any] | tuple[Any, ...]) -> str:
    values = mumps_or_values.icntl if isinstance(mumps_or_values, Mumps) else mumps_or_values
    return _display_values("ICNTL", list(values))


def display_cntl(mumps_or_values: Mumps | list[Any] | tuple[Any, ...]) -> str:
    values = mumps_or_values.cntl if isinstance(mumps_or_values, Mumps) else mumps_or_values
    return _display_values("CNTL", list(values))


def display_keep(mumps_or_values: Mumps | list[Any] | tuple[Any, ...]) -> str:
    values = mumps_or_values.keep if isinstance(mumps_or_values, Mumps) else mumps_or_values
    return _display_values("KEEP", list(values))


__all__ = [
    "DEFAULT_FORTRAN_COMMUNICATOR",
    "ICNTL_DEFAULT",
    "Mumps",
    "UnsupportedMumpsOperation",
    "default_icntl",
    "dense_matrix",
    "dense_rhs",
    "display_cntl",
    "display_icntl",
    "display_keep",
    "finalize",
    "finalize_unsafe",
    "get_rhs",
    "get_rhs_into",
    "get_schur_complement",
    "get_schur_complement_into",
    "get_sol",
    "get_sol_into",
    "has_det",
    "has_matrix",
    "has_rhs",
    "has_schur",
    "initialize",
    "invoke_mumps",
    "invoke_mumps_unsafe",
    "is_matrix_assembled",
    "is_matrix_distributed",
    "is_posdef",
    "is_rhs_dense",
    "is_sol_central",
    "is_symmetric",
    "mumps_det",
    "mumps_det_inplace",
    "mumps_factorize",
    "mumps_factorize_inplace",
    "mumps_schur_complement",
    "mumps_schur_complement_inplace",
    "mumps_select_inv",
    "mumps_select_inv_inplace",
    "mumps_solve",
    "mumps_solve_inplace",
    "provide_matrix",
    "provide_rhs",
    "provide_rhs_dense",
    "provide_rhs_sparse",
    "set_cntl",
    "set_error_stream",
    "set_diagnostics_stream",
    "set_icntl",
    "set_info_stream",
    "set_job",
    "set_keep",
    "set_print_level",
    "set_save_dir",
    "set_save_prefix",
    "set_schur_centralized_by_column",
    "sparse_matrix",
    "sparse_rhs",
    "suppress_display",
    "suppress_printing",
    "toggle_display",
    "toggle_null_pivot",
    "toggle_printing",
    "transpose_mumps",
]
