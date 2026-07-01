"""Generate v5 Julia parity fixtures for diagonal 3D FDFD boundary coverage.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_fdfd_3d_diagonal_v5_fixtures.jl'

The cases stay small enough for dense MAT storage while covering diagonal 3D
branches that v4 left unpinned: nonzero PML in all directions, numeric Bloch
phases, mixed non-periodic boundary conditions, and low-level SC-PML.
"""

using MAT
using MESTI
using SparseArrays

const OUT_PATH = joinpath(@__DIR__, "fdfd_3d_diagonal_v5_boundaries.mat")

function patterned_epsilon(nx, ny, nz, base)
    epsilon = zeros(ComplexF64, nx, ny, nz)
    for iz in 1:nz
        for iy in 1:ny
            for ix in 1:nx
                epsilon[ix, iy, iz] = base + 0.071 * ix + 0.017 * iy + 0.009 * iz
            end
        end
    end
    return epsilon
end

function pml_pair(low, high)
    return [PML(low), PML(high)]
end

function record_case!(
    payload,
    prefix,
    epsilon_xx,
    epsilon_yy,
    epsilon_zz,
    k0dx,
    xBC,
    yBC,
    zBC,
    xPML,
    yPML,
    zPML,
    use_UPML,
)
    A, is_symmetric_A, xPML_out, yPML_out, zPML_out = mesti_build_fdfd_matrix(
        epsilon_xx,
        epsilon_yy,
        epsilon_zz,
        k0dx,
        xBC,
        yBC,
        zBC,
        xPML,
        yPML,
        zPML,
        use_UPML,
    )

    payload["$(prefix)_epsilon_xx"] = epsilon_xx
    payload["$(prefix)_epsilon_yy"] = epsilon_yy
    payload["$(prefix)_epsilon_zz"] = epsilon_zz
    payload["$(prefix)_k0dx"] = k0dx
    payload["$(prefix)_xBC"] = xBC
    payload["$(prefix)_yBC"] = yBC
    payload["$(prefix)_zBC"] = zBC
    payload["$(prefix)_use_UPML"] = use_UPML
    payload["$(prefix)_xPML_low_npixels"] = xPML_out[1].npixels
    payload["$(prefix)_xPML_high_npixels"] = xPML_out[2].npixels
    payload["$(prefix)_yPML_low_npixels"] = yPML_out[1].npixels
    payload["$(prefix)_yPML_high_npixels"] = yPML_out[2].npixels
    payload["$(prefix)_zPML_low_npixels"] = zPML_out[1].npixels
    payload["$(prefix)_zPML_high_npixels"] = zPML_out[2].npixels
    payload["$(prefix)_is_symmetric_A"] = is_symmetric_A
    payload["$(prefix)_A_shape"] = collect(size(A))
    payload["$(prefix)_A_nnz"] = nnz(A)
    payload["$(prefix)_A_dense"] = Array(A)
    println("Recorded ", prefix, " with A size ", size(A), " and nnz ", nnz(A))
end

function main()
    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_fdfd_3d_diagonal_v5_fixtures.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "V5 diagonal 3D FDFD boundary/PML/Bloch parity bundle.",
    )

    record_case!(
        payload,
        "pml",
        patterned_epsilon(3, 3, 3, 1.00),
        patterned_epsilon(3, 3, 3, 1.17),
        patterned_epsilon(3, 3, 3, 1.34),
        0.71,
        "periodic",
        "periodic",
        "periodic",
        pml_pair(1, 1),
        pml_pair(1, 1),
        pml_pair(1, 1),
        true,
    )

    record_case!(
        payload,
        "bloch",
        patterned_epsilon(3, 3, 2, 0.94),
        patterned_epsilon(3, 3, 2, 1.11),
        patterned_epsilon(3, 3, 2, 1.28),
        0.83,
        0.37,
        -0.29,
        "periodic",
        pml_pair(0, 0),
        pml_pair(0, 0),
        pml_pair(0, 0),
        true,
    )

    record_case!(
        payload,
        "mixed_bc",
        patterned_epsilon(2, 3, 2, 1.02),
        patterned_epsilon(3, 3, 2, 1.19),
        patterned_epsilon(3, 3, 2, 1.36),
        0.67,
        "PMC",
        "PECPMC",
        "PMCPEC",
        pml_pair(0, 0),
        pml_pair(0, 0),
        pml_pair(0, 0),
        true,
    )

    record_case!(
        payload,
        "sc_pml",
        patterned_epsilon(3, 3, 3, 0.98),
        patterned_epsilon(3, 3, 3, 1.15),
        patterned_epsilon(3, 3, 3, 1.32),
        0.74,
        "periodic",
        "periodic",
        "periodic",
        pml_pair(1, 1),
        pml_pair(1, 1),
        pml_pair(1, 1),
        false,
    )

    matwrite(OUT_PATH, payload)
    println("Wrote ", OUT_PATH)
end

main()
