Here is the practical step-by-step guide for running [simu_test_2D_TM.py](<D:/BaiduSyncdisk/Projects/Q project/Simulation/python/simu_test_2D_TM.py>) in WSL with a folder containing `epsilon.mat`.

This file is prepared for a student who is new to the command line.

**1. Convert The Folder Path**
Windows path:

```text
G:\Data\Q_Project\Simulation\tilt\Ws30 Ls7.5\Structure 1
```

becomes WSL path:

```text
/mnt/g/Data/Q_Project/Simulation/tilt/Ws30 Ls7.5/Structure 1
```

In general:

```text
C:\... -> /mnt/c/...
D:\... -> /mnt/d/...
G:\... -> /mnt/g/...
```

**2. Open PowerShell**
From Windows PowerShell, run commands through WSL using `bash -ic`, which loads the WSL environment correctly.

**3. Check The Python Environment**
Use the WSL base Python path that already worked with `h5py` and `mumpspy`:

```powershell
wsl.exe --user lky -- bash -ic 'python -c "import numpy, scipy, h5py, mumpspy; print(\"Python/MUMPS environment OK\")"'
```

If this prints `Python/MUMPS environment OK`, continue.

**4. Run The Simulation**
Conservative template. This omits `--nrhs`, so sparse RHS solves use the
memory-aware default batch size:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project/Simulation/python" && python simu_test_2D_TM.py --root "<WSL_FOLDER_CONTAINING_EPSILON>" --input epsilon.mat --output-dir "<WSL_OUTPUT_FOLDER>" --solver mumpspy'
```

For your tested small case, `--nrhs 95` was faster and close to Julia timing,
but it used more peak memory:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project/Simulation/python" && python simu_test_2D_TM.py --root "/mnt/g/Data/Q_Project/Simulation/tilt/Ws30 Ls7.5/Structure 1" --input epsilon.mat --output-dir "/mnt/g/Data/Q_Project/Simulation/tilt/Ws30 Ls7.5/Structure 1" --solver mumpspy --nrhs 95'
```

**5. Optional: Record Time And Memory**
Use `/usr/bin/time -v`:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project/Simulation/python" && /usr/bin/time -v python simu_test_2D_TM.py --root "/mnt/g/Data/Q_Project/Simulation/tilt/Ws30 Ls7.5/Structure 1" --input epsilon.mat --output-dir "/mnt/g/Data/Q_Project/Simulation/tilt/Ws30 Ls7.5/Structure 1" --solver mumpspy --nrhs 95'
```

**6. Expected Outputs**
The script writes these files into `--output-dir`:

```text
py_TM_mscaepsilon.mat
py_Ex_eigen_epsilon.mat
```

`py_TM_mscaepsilon.mat` contains `t`.

`py_Ex_eigen_epsilon.mat` contains `Ex`.

Small notes:

- `--solver mumpspy` is the stable real-data route tested in WSL base Python.
- If you omit `--nrhs`, the solver chooses a conservative memory-aware sparse
  RHS batch width. Use an explicit value such as `--nrhs 95` only when you want
  to trade more memory for fewer solve calls.
- Python solves with double precision MUMPS bindings. If you compare against
  Julia real-data output at tight tolerance, generate the Julia reference with
  `opts.use_single_precision_MUMPS = false`; Julia's default MUMPS path uses
  single precision and differs at about the `1e-3` level on the tested small
  real case.

**7. Run The cuDSS/MUMPS APF Benchmark Test**

Use this section to reproduce the solver comparison table from
`tests/compare_matrix_memory_usage.py`, including SciPy factorize-and-solve,
MUMPS APF, cuDSS factorize-and-solve, and cuDSS APF.

Important: use `bash -ic`, not `bash -lc`. The interactive shell loads the
oneAPI/MPI/MKL/MUMPS library paths needed by `mumpspy`; without it, the
`mumps_apf` row may fail because `libmpi.so` cannot be found.

First, check that the benchmark environment can import the required packages:

```powershell
wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project/Simulation/python" && /home/lky/anaconda3/envs/optical_simulation/bin/python -c "import numpy, scipy, mumpspy, nvmath; print(\"benchmark environment OK\")"'
```

To reproduce the 900x900 benchmark case and record host peak memory, use the
project helper script:

```powershell
wsl.exe --user lky -- bash -ic 'bash "/mnt/d/BaiduSyncdisk/Projects/Q project/.codex/mesti_cudss_large_diagnostic/run_probe_900_mumps_cudss_fixed_env.sh"'
```

The helper script sets `LD_LIBRARY_PATH` carefully before running the benchmark:
the `nvidia-cublas-cu12` wheel library directory is placed before
`/usr/local/cuda-12.4/lib64`, while the Intel MPI, MKL, and MUMPS directories
remain available for `mumpspy`. This avoids the cuDSS loader error
`undefined symbol: cublasLtGetEnvironmentMode` and the MUMPS loader error
`cannot load MPI library`.

After the command finishes, read the concise table:

```powershell
Get-Content 'D:\BaiduSyncdisk\Projects\Q project\.codex\mesti_cudss_large_diagnostic\probe_900_mumps_fixed\mesti_solver_benchmark.txt'
```

Read the process-level peak memory and elapsed time:

```powershell
Select-String -Path 'D:\BaiduSyncdisk\Projects\Q project\.codex\mesti_cudss_large_diagnostic\probe_900_mumps_fixed\time.txt' -Pattern 'Maximum resident|Elapsed|Exit status|User time|System time'
```

Benchmark output files:

```text
.codex/mesti_cudss_large_diagnostic/probe_900_mumps_fixed/mesti_solver_benchmark.csv
.codex/mesti_cudss_large_diagnostic/probe_900_mumps_fixed/mesti_solver_benchmark.txt
.codex/mesti_cudss_large_diagnostic/probe_900_mumps_fixed/stdout.csv
.codex/mesti_cudss_large_diagnostic/probe_900_mumps_fixed/time.txt
```

Memory notes:

- The table's `Host Peak` column is Python `tracemalloc` memory for each
  backend row.
- The `/usr/bin/time -v` `Maximum resident set size` is the process-level host
  peak memory, including native solver allocations.
- `mumps_apf` uses `mumpspy` and the loaded MPI/MUMPS runtime. The separate
  `python-mumps` binding is supported for normal factorize-and-solve paths but
  is not the APF benchmark backend.
