"""cuDSS environment probing for the Python MESTI solver.

This module intentionally contains no solver dispatch yet.  The first cuDSS
slice only records whether the active Python environment has the CUDA/cuDSS
pieces needed for a later private compiled binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import ctypes.util
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sysconfig
import time
from typing import Any


@dataclass(frozen=True)
class CudssProbe:
    available: bool
    unavailable_reason: str | None
    binding_strategy: str
    cuda: dict[str, Any]
    cudss: dict[str, Any]
    device_array: dict[str, Any]
    compiler: dict[str, Any]
    python_packages: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "binding_strategy": self.binding_strategy,
            "cuda": self.cuda,
            "cudss": self.cudss,
            "device_array": self.device_array,
            "compiler": self.compiler,
            "python_packages": self.python_packages,
        }


def _safe_find_spec(module_name: str) -> Any:
    try:
        return importlib.util.find_spec(module_name)
    except (ImportError, ValueError, AttributeError):
        return None


def _safe_distribution_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_distribution_file(distribution_name: str, relative_path: str) -> Path | None:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    path = distribution.locate_file(relative_path)
    return Path(path) if path.exists() else None


def _run_command(args: list[str], timeout: float = 5.0) -> tuple[int | None, str, str]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, "", str(exc)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        if path is None:
            continue
        try:
            resolved = str(path.expanduser().resolve())
        except OSError:
            resolved = str(path.expanduser().absolute())
        key = resolved.lower() if os.name == "nt" else resolved
        if key not in seen:
            seen.add(key)
            unique.append(Path(resolved))
    return unique


def _existing_dirs(paths: list[Path]) -> list[Path]:
    return [path for path in _unique_paths(paths) if path.is_dir()]


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _cuda_roots() -> list[Path]:
    roots: list[Path] = []
    for name in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        path = _path_from_env(name)
        if path is not None:
            roots.append(path)
    for name, value in os.environ.items():
        if name.startswith("CUDA_PATH_") and value:
            roots.append(Path(value))
    if os.name == "nt":
        base = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        if base.is_dir():
            roots.extend(path for path in base.glob("v*") if path.is_dir())
    else:
        roots.extend([Path("/usr/local/cuda"), Path("/opt/cuda")])
    return _existing_dirs(roots)


def _nvidia_cudss_package_dirs() -> list[Path]:
    dirs: list[Path] = []
    for module_name in ("nvmath.bindings.cudss", "nvidia.cudss", "cudss"):
        spec = _safe_find_spec(module_name)
        if spec is None:
            continue
        if spec.submodule_search_locations:
            dirs.extend(Path(location) for location in spec.submodule_search_locations)
        elif spec.origin:
            dirs.append(Path(spec.origin).parent)
    for distribution_name in ("nvidia-cudss-cu12", "nvidia-cudss-cu13"):
        for relative_path in ("nvidia/cu12", "nvidia/cu13"):
            path = _safe_distribution_file(distribution_name, relative_path)
            if path is not None:
                dirs.append(path)
    return _existing_dirs(dirs)


def _cudss_roots() -> list[Path]:
    roots: list[Path] = []
    for name in ("CUDSS_PATH", "CUDSS_ROOT", "NVIDIA_CUDSS_ROOT"):
        path = _path_from_env(name)
        if path is not None:
            roots.append(path)
    roots.extend(_nvidia_cudss_package_dirs())
    roots.extend(_cuda_roots())
    return _existing_dirs(roots)


def _candidate_include_dirs() -> list[Path]:
    dirs: list[Path] = []
    for root in _cudss_roots():
        dirs.extend([root, root / "include", root / "include" / "cudss"])
    return _existing_dirs(dirs)


def _candidate_library_dirs() -> list[Path]:
    dirs: list[Path] = []
    for root in _cudss_roots():
        dirs.extend(
            [
                root,
                root / "bin",
                root / "lib",
                root / "lib64",
                root / "lib" / "x64",
                root / "lib" / "x86_64-linux-gnu",
            ]
        )
    return _existing_dirs(dirs)


def _find_headers() -> list[str]:
    headers: list[Path] = []
    for directory in _candidate_include_dirs():
        headers.extend(directory.glob("cudss.h"))
    return [str(path) for path in _unique_paths(headers)]


def _find_libraries() -> list[str]:
    libraries: list[Path] = []
    find_library = ctypes.util.find_library("cudss")
    if find_library:
        libraries.append(Path(find_library))
    patterns = ["cudss*.dll", "cudss*.lib"] if os.name == "nt" else ["libcudss.so*", "libcudss*.a", "libcudss*.dylib"]
    for directory in _candidate_library_dirs():
        for pattern in patterns:
            libraries.extend(path for path in directory.glob(pattern) if path.is_file())
    return [str(path) for path in _unique_paths(libraries)]


def _parse_cudss_header_version(header_path: str | None) -> str | None:
    if not header_path:
        return None
    try:
        text = Path(header_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    version_match = re.search(r"#\s*define\s+CUDSS_VERSION\s+([0-9]+)", text)
    if version_match:
        return version_match.group(1)
    parts: dict[str, str] = {}
    for name in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"#\s*define\s+CUDSS_VER_{name}\s+([0-9]+)", text)
        if match:
            parts[name] = match.group(1)
    if {"MAJOR", "MINOR"}.issubset(parts):
        return ".".join(parts[name] for name in ("MAJOR", "MINOR", "PATCH") if name in parts)
    return None


def _which(executable: str) -> str | None:
    found = shutil.which(executable)
    if found:
        return found
    for root in _cuda_roots():
        candidate = root / "bin" / executable
        if os.name == "nt" and candidate.suffix.lower() != ".exe":
            candidate = candidate.with_suffix(".exe")
        if candidate.is_file():
            return str(candidate)
    return None


def _nvcc_version(nvcc: str | None) -> str | None:
    if nvcc is None:
        return None
    returncode, stdout, stderr = _run_command([nvcc, "--version"])
    if returncode != 0:
        return None
    text = "\n".join(part for part in (stdout, stderr) if part)
    match = re.search(r"release\s+([0-9.]+)", text)
    return match.group(1) if match else text.splitlines()[-1] if text else None


def _gpu_from_nvidia_smi(nvidia_smi: str | None) -> tuple[list[str], str | None]:
    if nvidia_smi is None:
        return [], None
    returncode, stdout, _ = _run_command(
        [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
    )
    if returncode != 0 or not stdout:
        return [], None
    device_names: list[str] = []
    driver_version: str | None = None
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts and parts[0]:
            device_names.append(parts[0])
        if len(parts) > 1 and parts[1] and driver_version is None:
            driver_version = parts[1]
    return device_names, driver_version


def _gpu_from_cupy() -> tuple[list[str], str | None]:
    if _safe_find_spec("cupy") is None:
        return [], None
    try:
        import cupy
    except Exception:
        return [], None
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        return [], None
    names: list[str] = []
    for index in range(count):
        try:
            props = cupy.cuda.runtime.getDeviceProperties(index)
            raw_name = props.get("name", b"")
            names.append(raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name))
        except Exception:
            names.append(f"CUDA device {index}")
    try:
        runtime_version = str(cupy.cuda.runtime.runtimeGetVersion())
    except Exception:
        runtime_version = None
    return names, runtime_version


def _probe_cuda() -> dict[str, Any]:
    nvidia_smi = _which("nvidia-smi")
    nvcc = _which("nvcc")
    smi_names, driver_version = _gpu_from_nvidia_smi(nvidia_smi)
    cupy_names, runtime_version = _gpu_from_cupy()
    device_names = smi_names or cupy_names
    return {
        "platform": platform.platform(),
        "cuda_roots": [str(path) for path in _cuda_roots()],
        "nvcc": nvcc,
        "nvcc_version": _nvcc_version(nvcc),
        "nvidia_smi": nvidia_smi,
        "driver_version": driver_version,
        "runtime_version": runtime_version,
        "gpu_count": len(device_names),
        "device_names": device_names,
    }


def _probe_python_packages() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for module_name in (
        "nvmath",
        "nvmath.bindings",
        "nvmath.bindings.cudss",
        "cupy",
        "cuda",
        "cuda.bindings",
        "cudss",
        "nvidia.cudss",
    ):
        packages[module_name] = {"importable": _safe_find_spec(module_name) is not None}
    for distribution_name in (
        "nvmath-python",
        "cupy",
        "cuda-python",
        "nvidia-cudss-cu12",
        "nvidia-cudss-cu13",
        "cudss",
    ):
        version = _safe_distribution_version(distribution_name)
        if version is not None:
            packages[distribution_name] = {"installed": True, "version": version}
    return packages


def _probe_device_array(packages: dict[str, Any]) -> dict[str, Any]:
    candidates: list[str] = []
    if packages.get("cupy", {}).get("importable"):
        candidates.append("cupy")
    if packages.get("cuda.bindings", {}).get("importable") or packages.get("cuda", {}).get("importable"):
        candidates.append("cuda-python")
    candidates.append("compiled-extension")
    return {
        "selected": candidates[0],
        "candidates": candidates,
    }


def _probe_compiler() -> dict[str, Any]:
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(name: str, path: str | None, source: str) -> None:
        if not path:
            return
        key = path.lower() if os.name == "nt" else path
        if key in seen:
            return
        seen.add(key)
        candidates.append({"name": name, "path": path, "source": source})

    for env_name in ("CC", "CXX"):
        value = os.environ.get(env_name)
        if value:
            add(Path(value.split()[0]).name, shutil.which(value.split()[0]) or value.split()[0], env_name)
    cc = sysconfig.get_config_var("CC")
    if cc:
        add(Path(str(cc).split()[0]).name, shutil.which(str(cc).split()[0]) or str(cc).split()[0], "sysconfig.CC")
    for executable in ("cl", "gcc", "clang", "clang-cl", "nvcc"):
        add(executable, _which(executable), "PATH")

    return {
        "selected": candidates[0]["path"] if candidates else None,
        "candidates": candidates,
    }


def _probe_cudss() -> dict[str, Any]:
    headers = _find_headers()
    libraries = _find_libraries()
    python_binding = "nvmath.bindings.cudss" if _safe_find_spec("nvmath.bindings.cudss") is not None else None
    return {
        "python_binding": python_binding,
        "roots": [str(path) for path in _cudss_roots()],
        "headers": headers,
        "libraries": libraries,
        "header_version": _parse_cudss_header_version(headers[0] if headers else None),
    }


def _unavailable_reason(cuda: dict[str, Any], cudss: dict[str, Any], compiler: dict[str, Any], binding_strategy: str) -> str | None:
    if cuda["gpu_count"] < 1:
        return "no NVIDIA CUDA GPU was detected"
    if binding_strategy == "nvmath-bindings":
        return None
    if not cudss["headers"]:
        return "cuDSS header cudss.h was not found"
    if not cudss["libraries"]:
        return "cuDSS library was not found"
    if not compiler["selected"]:
        return "no C/C++ compiler was detected for a cuDSS extension build"
    return None


@lru_cache(maxsize=1)
def probe_environment() -> CudssProbe:
    """Return a cached, side-effect-free cuDSS environment probe."""

    cuda = _probe_cuda()
    cudss = _probe_cudss()
    compiler = _probe_compiler()
    python_packages = _probe_python_packages()
    device_array = _probe_device_array(python_packages)
    binding_strategy = "nvmath-bindings" if cudss["python_binding"] == "nvmath.bindings.cudss" else "compiled-extension"
    reason = _unavailable_reason(cuda, cudss, compiler, binding_strategy)
    return CudssProbe(
        available=reason is None,
        unavailable_reason=reason,
        binding_strategy=binding_strategy,
        cuda=cuda,
        cudss=cudss,
        device_array=device_array,
        compiler=compiler,
        python_packages=python_packages,
    )


def is_available() -> bool:
    return probe_environment().available


def require_available() -> CudssProbe:
    probe = probe_environment()
    if not probe.available:
        reason = probe.unavailable_reason or "cuDSS backend probe did not report availability"
        raise RuntimeError(
            "opts.solver='cudss' requires a usable cuDSS backend. "
            f"{reason}. Install nvmath-python with nvidia-cudss-cu12/cu13, or provide cuDSS headers/libs "
            "for a compiled extension fallback."
        )
    return probe


def _bool_option(value: Any, name: str) -> bool:
    import numpy as np

    if value is None:
        return False
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"opts.{name} must be a scalar")
    return bool(arr.reshape(-1)[0])


def _cudss_value_dtype(opts: Any) -> Any:
    import numpy as np

    if _bool_option(getattr(opts, "cudss_use_single_precision", None), "cudss_use_single_precision"):
        return np.complex64
    return np.complex128


def _cudss_hybrid_memory_requested(opts: Any) -> bool:
    if opts is None:
        return False
    return (
        _bool_option(getattr(opts, "cudss_use_hybrid_memory", None), "cudss_use_hybrid_memory")
        or getattr(opts, "cudss_hybrid_device_memory_limit", None) is not None
        or getattr(opts, "cudss_register_cuda_memory", None) is not None
    )


def _cudss_execution(opts: Any) -> Any:
    if not _cudss_hybrid_memory_requested(opts):
        return None

    try:
        from nvmath.sparse.advanced import ExecutionCUDA
        from nvmath.sparse.advanced._configuration import HybridMemoryModeOptions
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "cuDSS hybrid memory requires nvmath-python with "
            "ExecutionCUDA and HybridMemoryModeOptions support."
        ) from exc

    register_cuda_memory = getattr(opts, "cudss_register_cuda_memory", None)
    if register_cuda_memory is None:
        register_cuda_memory = True
    else:
        register_cuda_memory = _bool_option(register_cuda_memory, "cudss_register_cuda_memory")

    hybrid_options = HybridMemoryModeOptions(
        hybrid_memory_mode=True,
        hybrid_device_memory_limit=getattr(opts, "cudss_hybrid_device_memory_limit", None),
        register_cuda_memory=register_cuda_memory,
    )
    return ExecutionCUDA(hybrid_memory_mode_options=hybrid_options)


def _cudss_multithreading_lib() -> str | None:
    candidates: list[Path] = []
    for distribution_name, relative_root in (
        ("nvidia-cudss-cu12", "nvidia/cu12/lib"),
        ("nvidia-cudss-cu13", "nvidia/cu13/lib"),
    ):
        for filename in ("libcudss_mtlayer_gomp.so.0", "cudss_mtlayer_gomp.dll"):
            path = _safe_distribution_file(distribution_name, f"{relative_root}/{filename}")
            if path is not None:
                candidates.append(path)
    return str(candidates[0]) if candidates else None


def _cudss_direct_solver_options() -> Any:
    try:
        from nvmath.sparse.advanced import DirectSolverOptions
    except ImportError:
        return None
    multithreading_lib = _cudss_multithreading_lib()
    if multithreading_lib is None:
        return None
    return DirectSolverOptions(multithreading_lib=multithreading_lib)


def _as_csr(matrix: Any, dtype: Any) -> Any:
    import numpy as np
    from scipy import sparse

    if sparse.issparse(matrix):
        csr = matrix.astype(dtype, copy=False).tocsr()
    else:
        csr = sparse.csr_matrix(np.asarray(matrix, dtype=dtype))
    if csr.ndim != 2 or csr.shape[0] != csr.shape[1]:
        raise ValueError("cuDSS solve requires a square 2D matrix A.")
    if csr.indices.dtype != np.int32:
        csr.indices = csr.indices.astype(np.int32, copy=False)
    if csr.indptr.dtype != np.int32:
        csr.indptr = csr.indptr.astype(np.int32, copy=False)
    return csr


def _as_csr_complex128(matrix: Any) -> Any:
    import numpy as np

    return _as_csr(matrix, np.complex128)


def _as_rectangular_csr(matrix: Any, name: str, dtype: Any) -> Any:
    import numpy as np
    from scipy import sparse

    if sparse.issparse(matrix):
        csr = matrix.astype(dtype, copy=False).tocsr()
    else:
        arr = np.asarray(matrix, dtype=dtype)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        csr = sparse.csr_matrix(arr)
    if csr.ndim != 2:
        raise ValueError(f"cuDSS APF requires matrices.{name} to be 2D.")
    if csr.indices.dtype != np.int32:
        csr.indices = csr.indices.astype(np.int32, copy=False)
    if csr.indptr.dtype != np.int32:
        csr.indptr = csr.indptr.astype(np.int32, copy=False)
    return csr


def _as_rectangular_csr_complex128(matrix: Any, name: str) -> Any:
    import numpy as np

    return _as_rectangular_csr(matrix, name, np.complex128)


def _as_dense_rhs(matrix: Any, rows: int, dtype: Any) -> Any:
    import numpy as np
    from scipy import sparse

    if sparse.issparse(matrix):
        rhs = matrix.toarray()
    else:
        rhs = np.asarray(matrix, dtype=np.complex128)
    if rhs.ndim == 1:
        rhs = rhs[:, np.newaxis]
    if rhs.ndim != 2:
        raise ValueError("cuDSS solve requires a vector or 2D dense RHS B.")
    if rhs.shape[0] != rows:
        raise ValueError("cuDSS solve requires A row count to match B row count.")
    return np.asfortranarray(rhs, dtype=dtype)


def _as_dense_rhs_complex128(matrix: Any, rows: int) -> Any:
    import numpy as np

    return _as_dense_rhs(matrix, rows, np.complex128)


def cudss_solve(A: Any, B: Any, opts: Any, info: Any) -> Any:
    probe = require_available()
    if probe.binding_strategy != "nvmath-bindings":
        raise NotImplementedError(
            "cuDSS factorize-and-solve currently requires nvmath.bindings.cudss; "
            "compiled-extension fallback solving is not implemented yet."
        )

    from nvmath.sparse.advanced import DirectSolver

    dtype = _cudss_value_dtype(opts)
    A_csr = _as_csr(A, dtype)
    B_dense = _as_dense_rhs(B, A_csr.shape[0], dtype)

    solver = DirectSolver(A_csr, B_dense, options=_cudss_direct_solver_options(), execution=_cudss_execution(opts))
    try:
        start = time.perf_counter()
        solver.plan()
        if info is not None:
            info.timing_analyze = time.perf_counter() - start

        start = time.perf_counter()
        solver.factorize()
        if info is not None:
            info.timing_factorize = time.perf_counter() - start

        start = time.perf_counter()
        X = solver.solve()
        if info is not None:
            info.timing_solve = time.perf_counter() - start
    finally:
        solver.free()

    import numpy as np

    return np.asarray(X, dtype=np.complex128)


def _check_cuda_status(status: Any, operation: str, runtime: Any) -> None:
    if status != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed with CUDA status {status!s}.")


def cudss_apf(A: Any, B: Any, C: Any, opts: Any, transpose_B: bool, info: Any) -> Any:
    probe = require_available()
    if probe.binding_strategy != "nvmath-bindings":
        raise NotImplementedError(
            "cuDSS APF/Schur currently requires nvmath.bindings.cudss; "
            "compiled-extension fallback solving is not implemented yet."
        )
    try:
        from cuda.bindings import runtime
    except ImportError as exc:
        raise RuntimeError("cuDSS APF/Schur requires cuda-python for the Schur output buffer.") from exc

    import numpy as np
    from scipy import sparse
    from nvmath.bindings import cudss
    from nvmath.sparse.advanced import DirectSolver

    start_build = time.perf_counter()
    dtype = _cudss_value_dtype(opts)
    A_csr = _as_csr(A, dtype)
    B_block = _as_rectangular_csr(B, "B", dtype)
    C_block = B_block.transpose().tocsr() if transpose_B else _as_rectangular_csr(C, "C", dtype)

    n = A_csr.shape[0]
    if B_block.shape[0] != n:
        raise ValueError("cuDSS APF requires A row count to match B row count.")
    if C_block.shape[1] != n:
        raise ValueError("cuDSS APF requires C column count to match A row count.")

    n_in = B_block.shape[1]
    n_out = C_block.shape[0]
    n_schur = max(n_in, n_out)
    if n_schur == 0:
        return np.empty((n_out, n_in), dtype=np.complex128)
    if n_in < n_schur:
        B_block = sparse.hstack(
            [B_block, sparse.csr_matrix((n, n_schur - n_in), dtype=dtype)],
            format="csr",
        )
    if n_out < n_schur:
        C_block = sparse.vstack(
            [C_block, sparse.csr_matrix((n_schur - n_out, n), dtype=dtype)],
            format="csr",
        )

    zero = sparse.csr_matrix((n_schur, n_schur), dtype=dtype)
    K = sparse.bmat([[A_csr, B_block], [C_block, zero]], format="csr").astype(dtype, copy=False)
    if K.indices.dtype != np.int32:
        K.indices = K.indices.astype(np.int32, copy=False)
    if K.indptr.dtype != np.int32:
        K.indptr = K.indptr.astype(np.int32, copy=False)
    if info is not None:
        info.timing_build = time.perf_counter() - start_build

    # cuDSS Schur mode is limited to one matrix system and cannot be combined
    # with matching, COLAMD/BTF reorderings, multiblock factorization, or MGMN.
    # This path does not enable those cuDSS features.
    rhs = np.asfortranarray(np.zeros((K.shape[0], 1), dtype=dtype))
    solver = DirectSolver(K, rhs, options=_cudss_direct_solver_options(), execution=_cudss_execution(opts))
    schur_matrix = None
    schur_device = None
    try:
        schur_host = np.asfortranarray(np.zeros((n_schur, n_schur), dtype=dtype))
        status, schur_device = runtime.cudaMalloc(schur_host.nbytes)
        _check_cuda_status(status, "cudaMalloc for cuDSS Schur output", runtime)
        schur_matrix = cudss.matrix_create_dn(
            n_schur,
            n_schur,
            n_schur,
            schur_device,
            solver.cuda_value_type,
            cudss.Layout.COL_MAJOR,
        )

        mode = np.array([1], dtype=np.int32)
        schur_indices = np.zeros(K.shape[0], dtype=np.int32)
        schur_indices[n:] = 1
        cudss.config_set(solver.config_ptr, cudss.ConfigParam.SCHUR_MODE, mode.ctypes.data, mode.nbytes)
        cudss.data_set(
            solver.handle,
            solver.data_ptr,
            cudss.DataParam.USER_SCHUR_INDICES,
            schur_indices.ctypes.data,
            schur_indices.nbytes,
        )

        start_analyze = time.perf_counter()
        solver.plan()
        if info is not None:
            info.timing_analyze = time.perf_counter() - start_analyze

        start_factorize = time.perf_counter()
        solver.factorize()
        if info is not None:
            info.timing_factorize = time.perf_counter() - start_factorize

        start_solve = time.perf_counter()
        matrix_holder = np.array([schur_matrix], dtype=np.uintp)
        size_written = np.array([0], dtype=np.uint64)
        cudss.data_get(
            solver.handle,
            solver.data_ptr,
            cudss.DataParam.SCHUR_MATRIX,
            matrix_holder.ctypes.data,
            matrix_holder.nbytes,
            size_written.ctypes.data,
        )
        (status,) = runtime.cudaMemcpy(
            schur_host.ctypes.data,
            schur_device,
            schur_host.nbytes,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        _check_cuda_status(status, "cudaMemcpy for cuDSS Schur output", runtime)
        if info is not None:
            info.timing_solve = time.perf_counter() - start_solve
    finally:
        if schur_matrix is not None:
            cudss.matrix_destroy(schur_matrix)
        if schur_device is not None:
            (status,) = runtime.cudaFree(schur_device)
            _check_cuda_status(status, "cudaFree for cuDSS Schur output", runtime)
        solver.free()

    return -np.asarray(schur_host[:n_out, :n_in], dtype=np.complex128)
