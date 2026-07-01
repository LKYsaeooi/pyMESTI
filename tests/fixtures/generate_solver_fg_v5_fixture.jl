"""Generate a v5 Julia parity fixture for low-level solver FG.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_solver_fg_v5_fixture.jl'

The fixture is intentionally tiny.  It captures Julia's ``solver = "JULIA"``
``opts.method = "FG"`` path for an ordinary projected solve and for
``C = "transpose(B)"`` so Python can verify explicit SciPy/SuperLU FG parity
without relying on a Python MUMPS binding.
"""

using LinearAlgebra
using MAT
using MESTI
using SparseArrays

const OUT_PATH = joinpath(@__DIR__, "solver_fg_v5.mat")

function solve_case(A, B, C)
    matrices_fg = Matrices()
    matrices_fg.A = sparse(A)
    matrices_fg.B = sparse(B)
    matrices_fg.C = C == "transpose(B)" ? C : sparse(C)

    opts_fg = Opts()
    opts_fg.verbal = false
    opts_fg.solver = "JULIA"
    opts_fg.method = "FG"
    S_fg, info_fg = mesti_matrix_solver!(matrices_fg, opts_fg)

    matrices_fs = Matrices()
    matrices_fs.A = sparse(A)
    matrices_fs.B = sparse(B)
    matrices_fs.C = C == "transpose(B)" ? C : sparse(C)

    opts_fs = Opts()
    opts_fs.verbal = false
    opts_fs.solver = "JULIA"
    opts_fs.method = "factorize_and_solve"
    S_fs, info_fs = mesti_matrix_solver!(matrices_fs, opts_fs)

    return S_fg, S_fs, info_fg, info_fs
end

function main()
    A = ComplexF64[
        4.0+0.5im   0.7-0.2im   0.1+0.05im  0.0+0.0im
        0.2+0.1im   3.6+0.3im   0.8-0.1im  -0.2im
        0.0+0.2im  -0.4+0.1im   3.2+0.4im   0.6-0.3im
        0.3-0.1im   0.0+0.0im  -0.5+0.2im   2.8+0.6im
    ]
    B = ComplexF64[
        1.0+0.5im   0.0+0.0im   0.3-0.1im
        0.0+0.0im   1.2-0.4im   0.0+0.0im
       -0.7+0.2im   0.0+0.0im   1.1+0.3im
        0.0+0.0im  -0.5+0.6im   0.2+0.0im
    ]
    C = ComplexF64[
        0.5-0.2im   0.0+0.0im   1.0+0.1im   0.0+0.0im
        0.0+0.0im  -0.8+0.3im   0.0+0.0im   0.7-0.2im
    ]

    S_fg, S_fs, info_fg, info_fs = solve_case(A, B, C)
    S_t_fg, S_t_fs, info_t_fg, info_t_fs = solve_case(A, B, "transpose(B)")

    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "Simulation/python/tests/fixtures/generate_solver_fg_v5_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "V5 low-level solver Julia FG parity fixture.",
        "A" => A,
        "B" => B,
        "C" => C,
        "S_fg" => S_fg,
        "S_factorize_and_solve" => S_fs,
        "S_fg_singular_values" => svd(S_fg).S,
        "fg_info_solver" => info_fg.opts.solver,
        "fg_info_method" => info_fg.opts.method,
        "fs_info_solver" => info_fs.opts.solver,
        "fs_info_method" => info_fs.opts.method,
        "S_transpose_b_fg" => S_t_fg,
        "S_transpose_b_factorize_and_solve" => S_t_fs,
        "S_transpose_b_singular_values" => svd(S_t_fg).S,
        "transpose_b_fg_info_solver" => info_t_fg.opts.solver,
        "transpose_b_fg_info_method" => info_t_fg.opts.method,
        "transpose_b_fs_info_solver" => info_t_fs.opts.solver,
        "transpose_b_fs_info_method" => info_t_fs.opts.method,
    )

    matwrite(OUT_PATH, payload)
    println("wrote ", OUT_PATH)
    println("S_fg size=", size(S_fg), " method=", info_fg.opts.method)
    println("S_transpose_b_fg size=", size(S_t_fg), " method=", info_t_fg.opts.method)
end

main()
