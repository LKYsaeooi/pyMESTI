"""Generate a v6 Julia parity fixture for low-level MUMPS single precision.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_solver_mumps_single_precision_v6_fixture.jl'

The fixture is intentionally tiny.  It captures Julia's low-level
``opts.solver = "MUMPS"`` and ``opts.use_single_precision_MUMPS = true`` path
for ordinary factorize-and-solve, projected factorize-and-solve, and APF
projected solves so Python can verify the new ``mumpspy`` complex64 route
against a Julia MUMPS reference.
"""

using LinearAlgebra
using MAT
using MESTI
using SparseArrays

const OUT_PATH = joinpath(@__DIR__, "solver_mumps_single_precision_v6.mat")

function solve_mumps_case(A, B, C; method, use_single_precision)
    matrices = Matrices()
    matrices.A = sparse(A)
    matrices.B = sparse(B)
    if !isnothing(C)
        matrices.C = sparse(C)
    end

    opts = Opts()
    opts.verbal = false
    opts.verbal_solver = false
    opts.solver = "MUMPS"
    opts.method = method
    if method == "factorize_and_solve" && !isnothing(C)
        opts.nrhs = 2
    end
    opts.use_single_precision_MUMPS = use_single_precision
    opts.use_L0_threads = false

    S, info = mesti_matrix_solver!(matrices, opts)
    return S, info
end

function relative_error(reference, observed)
    denom = max(norm(reference), eps(Float64))
    return norm(observed - reference) / denom
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

    X_single, info_x_single = solve_mumps_case(
        A,
        B,
        nothing;
        method = "factorize_and_solve",
        use_single_precision = true,
    )
    X_double, info_x_double = solve_mumps_case(
        A,
        B,
        nothing;
        method = "factorize_and_solve",
        use_single_precision = false,
    )
    S_single_fs, info_s_single_fs = solve_mumps_case(
        A,
        B,
        C;
        method = "factorize_and_solve",
        use_single_precision = true,
    )
    S_double_fs, info_s_double_fs = solve_mumps_case(
        A,
        B,
        C;
        method = "factorize_and_solve",
        use_single_precision = false,
    )
    S_single_apf, info_s_single_apf = solve_mumps_case(
        A,
        B,
        C;
        method = "APF",
        use_single_precision = true,
    )
    S_double_apf, info_s_double_apf = solve_mumps_case(
        A,
        B,
        C;
        method = "APF",
        use_single_precision = false,
    )

    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "Simulation/python/tests/fixtures/generate_solver_mumps_single_precision_v6_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "V6 low-level solver Julia MUMPS single-precision parity fixture.",
        "A" => A,
        "B" => B,
        "C" => C,
        "use_single_precision_MUMPS" => true,
        "use_L0_threads" => false,
        "nrhs" => 2,
        "X_single_factorize_and_solve" => X_single,
        "X_double_factorize_and_solve" => X_double,
        "S_single_factorize_and_solve" => S_single_fs,
        "S_double_factorize_and_solve" => S_double_fs,
        "S_single_apf" => S_single_apf,
        "S_double_apf" => S_double_apf,
        "X_single_singular_values" => svd(X_single).S,
        "S_single_factorize_and_solve_singular_values" => svd(S_single_fs).S,
        "S_single_apf_singular_values" => svd(S_single_apf).S,
        "X_single_output_eltype" => string(eltype(X_single)),
        "S_single_factorize_and_solve_output_eltype" => string(eltype(S_single_fs)),
        "S_single_apf_output_eltype" => string(eltype(S_single_apf)),
        "X_single_vs_double_relerr" => relative_error(X_double, X_single),
        "S_factorize_and_solve_single_vs_double_relerr" => relative_error(S_double_fs, S_single_fs),
        "S_apf_single_vs_double_relerr" => relative_error(S_double_apf, S_single_apf),
        "X_single_info_solver" => info_x_single.opts.solver,
        "X_single_info_method" => info_x_single.opts.method,
        "X_single_info_use_single_precision_MUMPS" => info_x_single.opts.use_single_precision_MUMPS,
        "X_double_info_use_single_precision_MUMPS" => info_x_double.opts.use_single_precision_MUMPS,
        "S_single_fs_info_method" => info_s_single_fs.opts.method,
        "S_double_fs_info_method" => info_s_double_fs.opts.method,
        "S_single_apf_info_method" => info_s_single_apf.opts.method,
        "S_double_apf_info_method" => info_s_double_apf.opts.method,
        "S_single_apf_info_use_single_precision_MUMPS" => info_s_single_apf.opts.use_single_precision_MUMPS,
        "timing_X_single_total" => info_x_single.timing_total,
        "timing_S_single_fs_total" => info_s_single_fs.timing_total,
        "timing_S_single_apf_total" => info_s_single_apf.timing_total,
    )

    matwrite(OUT_PATH, payload)
    println("wrote ", OUT_PATH)
    println("X_single size=", size(X_single), " method=", info_x_single.opts.method)
    println("S_single_fs size=", size(S_single_fs), " method=", info_s_single_fs.opts.method)
    println("S_single_apf size=", size(S_single_apf), " method=", info_s_single_apf.opts.method)
    println("relative drifts: X=", payload["X_single_vs_double_relerr"],
            " FS=", payload["S_factorize_and_solve_single_vs_double_relerr"],
            " APF=", payload["S_apf_single_vs_double_relerr"])
end

main()
