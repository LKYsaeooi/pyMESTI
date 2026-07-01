"""Generate the Step 7 Julia parity fixture for 3D diagonal FDFD assembly.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_fdfd_3d_diagonal_julia_fixture.jl'

The fixture is intentionally tiny.  It locks down the first post-2D vectorial
slice: Yee-staggered 3D matrix assembly with diagonal tensor permittivity and
PEC boundaries, without exercising the still-deferred off-diagonal tensor,
channel, or 3D high-level solve paths.
"""

using MAT
using MESTI
using SparseArrays

const OUT_PATH = joinpath(@__DIR__, "fdfd_3d_diagonal_pec.mat")

function patterned_epsilon(nx, ny, nz, base)
    epsilon = zeros(ComplexF64, nx, ny, nz)
    for iz in 1:nz
        for iy in 1:ny
            for ix in 1:nx
                epsilon[ix, iy, iz] = base + 0.11 * ix + 0.013 * iy + 0.007 * iz
            end
        end
    end
    return epsilon
end

function main()
    # All three directions use PEC, which exercises Julia's Yee-grid staggering:
    # epsilon_xx is one x-site larger, epsilon_yy is one y-site larger, and
    # epsilon_zz is one z-site larger than the other electric components.
    epsilon_xx = patterned_epsilon(3, 2, 2, 1.0)
    epsilon_yy = patterned_epsilon(2, 3, 2, 1.5)
    epsilon_zz = patterned_epsilon(2, 2, 3, 2.0)
    k0dx = 0.73
    xBC = "PEC"
    yBC = "PEC"
    zBC = "PEC"
    xPML = [PML(0), PML(0)]
    yPML = [PML(0), PML(0)]
    zPML = [PML(0), PML(0)]
    use_UPML = true

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

    matwrite(
        OUT_PATH,
        Dict{String,Any}(
            "fixture_format" => 1,
            "generator" => "generate_fdfd_3d_diagonal_julia_fixture.jl",
            "julia_mesti_version" => "0.5.1",
            "epsilon_xx" => epsilon_xx,
            "epsilon_yy" => epsilon_yy,
            "epsilon_zz" => epsilon_zz,
            "k0dx" => k0dx,
            "xBC" => xBC,
            "yBC" => yBC,
            "zBC" => zBC,
            "use_UPML" => use_UPML,
            "xPML_low_npixels" => xPML_out[1].npixels,
            "xPML_high_npixels" => xPML_out[2].npixels,
            "yPML_low_npixels" => yPML_out[1].npixels,
            "yPML_high_npixels" => yPML_out[2].npixels,
            "zPML_low_npixels" => zPML_out[1].npixels,
            "zPML_high_npixels" => zPML_out[2].npixels,
            "is_symmetric_A" => is_symmetric_A,
            "A_shape" => collect(size(A)),
            "A_nnz" => nnz(A),
            "A_dense" => Array(A),
        ),
    )
    println("Wrote ", OUT_PATH, " with A size ", size(A), " and nnz ", nnz(A))
end

main()
