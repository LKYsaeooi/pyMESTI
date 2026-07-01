"""Generate v5 Julia parity fixtures for direct diagonal 3D ``mesti``.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti_3d_direct_v5_fixtures.jl'

The references use Julia's low-level 3D FDFD matrix followed by ``A \\ B``.
This matches the v3 dense-RHS fixture strategy and avoids Julia MESTI 0.5.1's
known high-level dense-B direct wrapper issue while still validating Python's
direct ``mesti`` wrapper, PML parsing, boundary parsing, and field reshaping.
"""

using MAT
using MESTI
using SparseArrays

const OUT_PATH = joinpath(@__DIR__, "mesti_3d_direct_v5_boundaries.mat")

function patterned_epsilon(nx, ny, nz, base)
    epsilon = zeros(ComplexF64, nx, ny, nz)
    for iz in 1:nz
        for iy in 1:ny
            for ix in 1:nx
                epsilon[ix, iy, iz] = base + 0.061 * ix + 0.014 * iy + 0.006 * iz
            end
        end
    end
    return epsilon
end

function pml_pair(low, high)
    return [PML(low), PML(high)]
end

function component_rhs(nx, ny, nz, nrhs, scale)
    out = zeros(ComplexF64, nx * ny * nz, nrhs)
    for col in 1:nrhs
        for iz in 1:nz
            for iy in 1:ny
                for ix in 1:nx
                    row = ix + (iy - 1) * nx + (iz - 1) * nx * ny
                    out[row, col] =
                        scale * (0.025 * ix - 0.012 * iy + 0.009 * iz + 0.031 * col) +
                        im * scale * (0.017 * ix + 0.021 * iy - 0.010 * iz + 0.026 * col)
                end
            end
        end
    end
    return out
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
        true,
    )

    nrhs = 2
    Bx = component_rhs(size(epsilon_xx)..., nrhs, 1.0)
    By = component_rhs(size(epsilon_yy)..., nrhs, -0.6)
    Bz = component_rhs(size(epsilon_zz)..., nrhs, 1.2)
    B = [Bx; By; Bz]
    X = A \ B

    nt_Ex = length(epsilon_xx)
    nt_Ey = length(epsilon_yy)
    nt_Ez = length(epsilon_zz)
    Ex = reshape(X[1:nt_Ex, :], size(epsilon_xx)..., nrhs)
    Ey = reshape(X[(nt_Ex + 1):(nt_Ex + nt_Ey), :], size(epsilon_yy)..., nrhs)
    Ez = reshape(X[(nt_Ex + nt_Ey + 1):(nt_Ex + nt_Ey + nt_Ez), :], size(epsilon_zz)..., nrhs)

    payload["$(prefix)_epsilon_xx"] = epsilon_xx
    payload["$(prefix)_epsilon_yy"] = epsilon_yy
    payload["$(prefix)_epsilon_zz"] = epsilon_zz
    payload["$(prefix)_wavelength"] = 2 * pi / k0dx
    payload["$(prefix)_dx"] = 1.0
    payload["$(prefix)_xBC"] = xBC
    payload["$(prefix)_yBC"] = yBC
    payload["$(prefix)_zBC"] = zBC
    payload["$(prefix)_xPML_low_npixels"] = xPML_out[1].npixels
    payload["$(prefix)_xPML_high_npixels"] = xPML_out[2].npixels
    payload["$(prefix)_yPML_low_npixels"] = yPML_out[1].npixels
    payload["$(prefix)_yPML_high_npixels"] = yPML_out[2].npixels
    payload["$(prefix)_zPML_low_npixels"] = zPML_out[1].npixels
    payload["$(prefix)_zPML_high_npixels"] = zPML_out[2].npixels
    payload["$(prefix)_A_shape"] = collect(size(A))
    payload["$(prefix)_A_nnz"] = nnz(A)
    payload["$(prefix)_is_symmetric_A"] = is_symmetric_A
    payload["$(prefix)_B"] = B
    payload["$(prefix)_field_Ex"] = Ex
    payload["$(prefix)_field_Ey"] = Ey
    payload["$(prefix)_field_Ez"] = Ez
    println("Recorded ", prefix, " with A size ", size(A), ", nnz ", nnz(A), ", and ", nrhs, " RHS")
end

function main()
    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_mesti_3d_direct_v5_fixtures.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "V5 direct diagonal 3D dense-RHS fixture bundle for PML, Bloch, and mixed boundaries.",
    )

    record_case!(
        payload,
        "pml",
        patterned_epsilon(3, 3, 3, 1.00),
        patterned_epsilon(3, 3, 3, 1.16),
        patterned_epsilon(3, 3, 3, 1.32),
        0.72,
        "periodic",
        "periodic",
        "periodic",
        pml_pair(1, 1),
        pml_pair(1, 1),
        pml_pair(1, 1),
    )

    record_case!(
        payload,
        "bloch",
        patterned_epsilon(3, 3, 2, 0.97),
        patterned_epsilon(3, 3, 2, 1.13),
        patterned_epsilon(3, 3, 2, 1.29),
        0.81,
        0.37,
        -0.29,
        0.23,
        pml_pair(0, 0),
        pml_pair(0, 0),
        pml_pair(0, 0),
    )

    record_case!(
        payload,
        "mixed_bc",
        patterned_epsilon(2, 3, 2, 1.03),
        patterned_epsilon(3, 3, 2, 1.19),
        patterned_epsilon(3, 3, 2, 1.35),
        0.69,
        "PMC",
        "PECPMC",
        "PMCPEC",
        pml_pair(0, 0),
        pml_pair(0, 0),
        pml_pair(0, 0),
    )

    matwrite(OUT_PATH, payload)
    println("Wrote ", OUT_PATH)
end

main()
