"""Generate v5 Julia parity fixtures for direct ``mesti`` option handling.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti_direct_options_v5_fixture.jl'

The references use Julia's low-level FDFD matrix followed by direct ``A \\ B``
solves.  This keeps the fixture independent of Julia MESTI 0.5.1's known
high-level dense-B direct wrapper issue while pinning option-surface behavior:
SC-PML, Bloch convenience phases, PML exclusion, prefactor application, and
``C = "transpose(B)"`` projection semantics.
"""

using MAT
using MESTI
using SparseArrays

const OUT_PATH = joinpath(@__DIR__, "mesti_direct_options_v5.mat")

function patterned_epsilon_2d(ny, nz, base)
    epsilon = zeros(ComplexF64, ny, nz)
    for iz in 1:nz
        for iy in 1:ny
            epsilon[iy, iz] = base + 0.041 * iy + 0.013 * iz
        end
    end
    return epsilon
end

function patterned_epsilon_3d(nx, ny, nz, base)
    epsilon = zeros(ComplexF64, nx, ny, nz)
    for iz in 1:nz
        for iy in 1:ny
            for ix in 1:nx
                epsilon[ix, iy, iz] = base + 0.052 * ix + 0.017 * iy + 0.009 * iz
            end
        end
    end
    return epsilon
end

function rhs_2d(ny, nz, nrhs, scale)
    out = zeros(ComplexF64, ny * nz, nrhs)
    for col in 1:nrhs
        for iz in 1:nz
            for iy in 1:ny
                row = iy + (iz - 1) * ny
                out[row, col] =
                    scale * (0.023 * iy - 0.011 * iz + 0.031 * col) +
                    im * scale * (0.017 * iy + 0.019 * iz - 0.007 * col)
            end
        end
    end
    return out
end

function component_rhs(nx, ny, nz, nrhs, scale)
    out = zeros(ComplexF64, nx * ny * nz, nrhs)
    for col in 1:nrhs
        for iz in 1:nz
            for iy in 1:ny
                for ix in 1:nx
                    row = ix + (iy - 1) * nx + (iz - 1) * nx * ny
                    out[row, col] =
                        scale * (0.021 * ix - 0.014 * iy + 0.008 * iz + 0.027 * col) +
                        im * scale * (0.013 * ix + 0.018 * iy - 0.010 * iz + 0.022 * col)
                end
            end
        end
    end
    return out
end

function direct_matrix(nrows, ncols)
    out = zeros(ComplexF64, nrows, ncols)
    for col in 1:ncols
        for row in 1:nrows
            out[row, col] = 0.009 * row - 0.004 * col + im * (0.003 * row + 0.005 * col)
        end
    end
    return out
end

function pml_pair(low, high)
    return [PML(low), PML(high)]
end

function record_2d_sc_pml!(payload)
    epsilon_xx = patterned_epsilon_2d(4, 5, 1.02)
    k0dx = 0.67
    dx = 1.0
    yPML = pml_pair(1, 1)
    zPML = pml_pair(1, 1)
    A, is_symmetric_A, yPML_out, zPML_out = mesti_build_fdfd_matrix(
        epsilon_xx,
        k0dx,
        "PEC",
        "PEC",
        yPML,
        zPML,
        false,
    )
    nrhs = 2
    B = rhs_2d(size(epsilon_xx)..., nrhs, 1.0)
    X = A \ B
    prefactor = 1.20 - 0.35im
    field_full = reshape(prefactor * X, size(epsilon_xx)..., nrhs)
    field_trim = field_full[
        (yPML_out[1].npixels + 1):(size(epsilon_xx, 1) - yPML_out[2].npixels),
        (zPML_out[1].npixels + 1):(size(epsilon_xx, 2) - zPML_out[2].npixels),
        :,
    ]
    D = direct_matrix(nrhs, nrhs)
    projection_transpose_B = prefactor * (transpose(B) * X) - D

    payload["twod_epsilon_xx"] = epsilon_xx
    payload["twod_wavelength"] = 2 * pi * dx / k0dx
    payload["twod_dx"] = dx
    payload["twod_yPML_low_npixels"] = yPML_out[1].npixels
    payload["twod_yPML_high_npixels"] = yPML_out[2].npixels
    payload["twod_zPML_low_npixels"] = zPML_out[1].npixels
    payload["twod_zPML_high_npixels"] = zPML_out[2].npixels
    payload["twod_is_symmetric_A"] = is_symmetric_A
    payload["twod_B"] = B
    payload["twod_D"] = D
    payload["twod_prefactor"] = prefactor
    payload["twod_field_full"] = field_full
    payload["twod_field_trim"] = field_trim
    payload["twod_projection_transpose_B"] = projection_transpose_B
    println("Recorded twod SC-PML direct options with A size ", size(A), " and nnz ", nnz(A))
end

function record_2d_bloch!(payload)
    epsilon_xx = patterned_epsilon_2d(3, 4, 0.96)
    k0dx = 0.58
    dx = 0.75
    y_phase = 0.31
    z_phase = -0.27
    A, is_symmetric_A, _, _ = mesti_build_fdfd_matrix(
        epsilon_xx,
        k0dx,
        y_phase,
        z_phase,
        pml_pair(0, 0),
        pml_pair(0, 0),
        true,
    )
    nrhs = 2
    B = rhs_2d(size(epsilon_xx)..., nrhs, -0.8)
    X = A \ B
    field = reshape(X, size(epsilon_xx)..., nrhs)

    payload["twod_bloch_epsilon_xx"] = epsilon_xx
    payload["twod_bloch_wavelength"] = 2 * pi * dx / k0dx
    payload["twod_bloch_dx"] = dx
    payload["twod_bloch_ky_B"] = y_phase / (size(epsilon_xx, 1) * dx)
    payload["twod_bloch_kz_B"] = z_phase / (size(epsilon_xx, 2) * dx)
    payload["twod_bloch_is_symmetric_A"] = is_symmetric_A
    payload["twod_bloch_B"] = B
    payload["twod_bloch_field"] = field
    println("Recorded twod Bloch convenience direct options with A size ", size(A), " and nnz ", nnz(A))
end

function record_3d_sc_pml!(payload)
    epsilon_xx = patterned_epsilon_3d(3, 3, 3, 1.01)
    epsilon_yy = patterned_epsilon_3d(3, 3, 3, 1.18)
    epsilon_zz = patterned_epsilon_3d(3, 3, 3, 1.35)
    k0dx = 0.62
    dx = 1.0
    xPML = pml_pair(1, 1)
    yPML = pml_pair(1, 1)
    zPML = pml_pair(1, 1)
    A, is_symmetric_A, xPML_out, yPML_out, zPML_out = mesti_build_fdfd_matrix(
        epsilon_xx,
        epsilon_yy,
        epsilon_zz,
        k0dx,
        "periodic",
        "periodic",
        "periodic",
        xPML,
        yPML,
        zPML,
        false,
    )
    nrhs = 2
    Bx = component_rhs(size(epsilon_xx)..., nrhs, 1.0)
    By = component_rhs(size(epsilon_yy)..., nrhs, -0.7)
    Bz = component_rhs(size(epsilon_zz)..., nrhs, 1.1)
    B = [Bx; By; Bz]
    X = A \ B
    prefactor = -0.85 + 0.25im
    X = prefactor * X
    nt_Ex = length(epsilon_xx)
    nt_Ey = length(epsilon_yy)
    nt_Ez = length(epsilon_zz)
    Ex_full = reshape(X[1:nt_Ex, :], size(epsilon_xx)..., nrhs)
    Ey_full = reshape(X[(nt_Ex + 1):(nt_Ex + nt_Ey), :], size(epsilon_yy)..., nrhs)
    Ez_full = reshape(X[(nt_Ex + nt_Ey + 1):(nt_Ex + nt_Ey + nt_Ez), :], size(epsilon_zz)..., nrhs)
    Ex_trim = Ex_full[
        (xPML_out[1].npixels + 1):(size(epsilon_xx, 1) - xPML_out[2].npixels),
        (yPML_out[1].npixels + 1):(size(epsilon_xx, 2) - yPML_out[2].npixels),
        (zPML_out[1].npixels + 1):(size(epsilon_xx, 3) - zPML_out[2].npixels),
        :,
    ]
    Ey_trim = Ey_full[
        (xPML_out[1].npixels + 1):(size(epsilon_yy, 1) - xPML_out[2].npixels),
        (yPML_out[1].npixels + 1):(size(epsilon_yy, 2) - yPML_out[2].npixels),
        (zPML_out[1].npixels + 1):(size(epsilon_yy, 3) - zPML_out[2].npixels),
        :,
    ]
    Ez_trim = Ez_full[
        (xPML_out[1].npixels + 1):(size(epsilon_zz, 1) - xPML_out[2].npixels),
        (yPML_out[1].npixels + 1):(size(epsilon_zz, 2) - yPML_out[2].npixels),
        (zPML_out[1].npixels + 1):(size(epsilon_zz, 3) - zPML_out[2].npixels),
        :,
    ]

    payload["threed_epsilon_xx"] = epsilon_xx
    payload["threed_epsilon_yy"] = epsilon_yy
    payload["threed_epsilon_zz"] = epsilon_zz
    payload["threed_wavelength"] = 2 * pi * dx / k0dx
    payload["threed_dx"] = dx
    payload["threed_xPML_low_npixels"] = xPML_out[1].npixels
    payload["threed_xPML_high_npixels"] = xPML_out[2].npixels
    payload["threed_yPML_low_npixels"] = yPML_out[1].npixels
    payload["threed_yPML_high_npixels"] = yPML_out[2].npixels
    payload["threed_zPML_low_npixels"] = zPML_out[1].npixels
    payload["threed_zPML_high_npixels"] = zPML_out[2].npixels
    payload["threed_is_symmetric_A"] = is_symmetric_A
    payload["threed_B"] = B
    payload["threed_prefactor"] = prefactor
    payload["threed_field_Ex_trim"] = Ex_trim
    payload["threed_field_Ey_trim"] = Ey_trim
    payload["threed_field_Ez_trim"] = Ez_trim
    println("Recorded threed SC-PML direct options with A size ", size(A), " and nnz ", nnz(A))
end

function main()
    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_mesti_direct_options_v5_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "V5 direct mesti option-surface fixture bundle.",
    )
    record_2d_sc_pml!(payload)
    record_2d_bloch!(payload)
    record_3d_sc_pml!(payload)
    matwrite(OUT_PATH, payload)
    println("Wrote ", OUT_PATH)
end

main()
