"""Explicit unsupported companion for Julia ``MPI/hybrid_mpi.jl``.

The Julia script uses MPI ranks directly: non-root ranks enter raw MUMPS
analysis/factorization jobs, while the root rank builds a tiny random 3D
scattering system and calls ``mesti2s``.  The Python port does not expose raw
MUMPS C/MPI invocation or worker-rank orchestration, so this file records that
decision instead of pretending a single-process fallback is equivalent.

Run from ``Simulation/python`` to print the migration notes:

    python examples/hybrid_mpi.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesti import UnsupportedMumpsOperation


MIGRATION_NOTES = (
    "Use high-level Python mesti2s with solver='scipy' or a supported Python "
    "MUMPS binding for single-process correctness tests.",
    "For distributed MUMPS worker ranks, build an external mpi4py workflow "
    "around a Python MUMPS package that exposes the required MPI communicator "
    "and raw job lifecycle.",
    "Do not treat this stub as production MPI parity; it intentionally does "
    "not launch ranks, call invoke_mumps, or manage raw MUMPS job 1/2 workers.",
)


def hybrid_mpi_unsupported_reason() -> str:
    """Return the recorded Step 7 decision for the Julia hybrid MPI example."""

    return (
        "MPI/hybrid_mpi.jl depends on raw MUMPS C/MPI worker orchestration, "
        "which is intentionally unsupported in the Python port."
    )


def hybrid_mpi_migration_notes() -> tuple[str, ...]:
    """Return short migration notes for users who need real MPI behavior."""

    return MIGRATION_NOTES


def run_hybrid_mpi() -> None:
    """Raise the explicit unsupported error for the hybrid MPI example."""

    raise UnsupportedMumpsOperation(hybrid_mpi_unsupported_reason())


def main(*, stream: TextIO | None = None) -> int:
    """Print the unsupported decision and return a nonzero status code."""

    output = sys.stderr if stream is None else stream
    print(hybrid_mpi_unsupported_reason(), file=output)
    for index, note in enumerate(MIGRATION_NOTES, start=1):
        print(f"{index}. {note}", file=output)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
