"""Generate v5 Julia parity fixtures for direct off-diagonal 3D ``mesti``.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti_3d_direct_offdiagonal_v5_fixture.jl'

The references use Julia's low-level 3D FDFD matrix followed by ``A \\ B``.
This avoids Julia MESTI 0.5.1's known high-level dense-B direct wrapper issue
while validating Python's direct wrapper against off-diagonal tensor operators.
"""

using MAT
using MESTI
using SparseArrays

const OUT_PATH = joinpath(@__DIR__, "mesti_3d_direct_offdiagonal_v5.mat")

function patterned_epsilon(nx, ny, nz, base; imag_scale = 0.0)
    epsilon = zeros(ComplexF64, nx, ny, nz)
    for iz in 1:nz
        for iy in 1:ny
            for ix in 1:nx
                real_part = base + 0.067 * ix + 0.019 * iy + 0.011 * iz
                imag_part = imag_scale * (0.013 * ix - 0.007 * iy + 0.005 * iz)
                epsilon[ix, iy, iz] = real_part + 1im * imag_part
            end
        end
    end
    return epsilon
end

function patterned_offdiag(nx, ny, nz, base; imag_scale = 0.0)
    epsilon = zeros(ComplexF64, nx, ny, nz)
    for iz in 1:nz
        for iy in 1:ny
            for ix in 1:nx
                real_part = base + 0.009 * ix - 0.004 * iy + 0.006 * iz
                imag_part = imag_scale * (0.003 * ix + 0.005 * iy - 0.002 * iz)
                epsilon[ix, iy, iz] = real_part + 1im * imag_part
            end
        end
    end
    return epsilon
end

function pml_pair(low, high)
    return [PML(low), PML(high)]
end

function offdiag_shapes(epsilon_xx, epsilon_yy, epsilon_zz)
    nx_Ex, ny_Ex, nz_Ex = size(epsilon_xx)
    nx_Ey, ny_Ey, nz_Ey = size(epsilon_yy)
    nx_Ez, ny_Ez, nz_Ez = size(epsilon_zz)
    return Dict(
        "xy" => (nx_Ez, ny_Ez, nz_Ex),
        "xz" => (nx_Ey, ny_Ex, nz_Ey),
        "yx" => (nx_Ez, ny_Ez, nz_Ey),
        "yz" => (nx_Ey, ny_Ex, nz_Ex),
        "zx" => (nx_Ey, ny_Ez, nz_Ey),
        "zy" => (nx_Ez, ny_Ex, nz_Ex),
    )
end

function hermitian_offdiagonal(epsilon_xx, epsilon_yy, epsilon_zz; scale = 1.0)
    shapes = offdiag_shapes(epsilon_xx, epsilon_yy, epsilon_zz)
    epsilon_xy = patterned_offdiag(shapes["xy"]..., 0.020 * scale)
    epsilon_xz = patterned_offdiag(shapes["xz"]..., -0.014 * scale)
    epsilon_yz = patterned_offdiag(shapes["yz"]..., 0.011 * scale)
    return (
        epsilon_xy,
        epsilon_xz,
        conj.(epsilon_xy),
        epsilon_yz,
        conj.(epsilon_xz),
        conj.(epsilon_yz),
    )
end

function lossy_offdiagonal(epsilon_xx, epsilon_yy, epsilon_zz)
    shapes = offdiag_shapes(epsilon_xx, epsilon_yy, epsilon_zz)
    return (
        patterned_offdiag(shapes["xy"]..., 0.023; imag_scale = 0.6),
        patterned_offdiag(shapes["xz"]..., -0.019; imag_scale = 0.5),
        patterned_offdiag(shapes["yx"]..., 0.017; imag_scale = -0.4),
        patterned_offdiag(shapes["yz"]..., 0.013; imag_scale = 0.3),
        patterned_offdiag(shapes["zx"]..., -0.010; imag_scale = -0.2),
        patterned_offdiag(shapes["zy"]..., 0.007; imag_scale = 0.7),
    )
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

function projection_matrix(noutputs, ntotal)
    out = zeros(ComplexF64, noutputs, ntotal)
    for row in 1:noutputs
        for col in 1:ntotal
            out[row, col] =
                0.005 * row + 0.0025 * col +
                im * (0.004 * row - 0.0015 * col)
        end
    end
    return out
end

function direct_matrix(noutputs, nrhs)
    out = zeros(ComplexF64, noutputs, nrhs)
    for row in 1:noutputs
        for col in 1:nrhs
            out[row, col] = 0.011 * row - 0.003 * col + im * (0.002 * row + 0.005 * col)
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
    epsilon_xy,
    epsilon_xz,
    epsilon_yx,
    epsilon_yz,
    epsilon_zx,
    epsilon_zy,
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
        epsilon_xy,
        epsilon_xz,
        epsilon_yx,
        epsilon_yy,
        epsilon_yz,
        epsilon_zx,
        epsilon_zy,
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
    noutputs = 3
    nt_Ex = length(epsilon_xx)
    nt_Ey = length(epsilon_yy)
    nt_Ez = length(epsilon_zz)
    ntotal = nt_Ex + nt_Ey + nt_Ez
    Bx = component_rhs(size(epsilon_xx)..., nrhs, 1.0)
    By = component_rhs(size(epsilon_yy)..., nrhs, -0.6)
    Bz = component_rhs(size(epsilon_zz)..., nrhs, 1.2)
    B = [Bx; By; Bz]
    C = projection_matrix(noutputs, ntotal)
    D = direct_matrix(noutputs, nrhs)
    X = A \ B

    Ex = reshape(X[1:nt_Ex, :], size(epsilon_xx)..., nrhs)
    Ey = reshape(X[(nt_Ex + 1):(nt_Ex + nt_Ey), :], size(epsilon_yy)..., nrhs)
    Ez = reshape(X[(nt_Ex + nt_Ey + 1):ntotal, :], size(epsilon_zz)..., nrhs)
    projection_with_D = C * X - D

    payload["$(prefix)_epsilon_xx"] = epsilon_xx
    payload["$(prefix)_epsilon_yy"] = epsilon_yy
    payload["$(prefix)_epsilon_zz"] = epsilon_zz
    payload["$(prefix)_epsilon_xy"] = epsilon_xy
    payload["$(prefix)_epsilon_xz"] = epsilon_xz
    payload["$(prefix)_epsilon_yx"] = epsilon_yx
    payload["$(prefix)_epsilon_yz"] = epsilon_yz
    payload["$(prefix)_epsilon_zx"] = epsilon_zx
    payload["$(prefix)_epsilon_zy"] = epsilon_zy
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
    payload["$(prefix)_C"] = C
    payload["$(prefix)_D"] = D
    payload["$(prefix)_field_Ex"] = Ex
    payload["$(prefix)_field_Ey"] = Ey
    payload["$(prefix)_field_Ez"] = Ez
    payload["$(prefix)_projection_with_D"] = projection_with_D
    println("Recorded ", prefix, " with A size ", size(A), ", nnz ", nnz(A), ", and ", nrhs, " RHS")
end

function main()
    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_mesti_3d_direct_offdiagonal_v5_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "V5 direct off-diagonal 3D dense-RHS/projection fixture bundle.",
    )

    eps_xx = patterned_epsilon(2, 2, 2, 1.00)
    eps_yy = patterned_epsilon(2, 2, 2, 1.18)
    eps_zz = patterned_epsilon(2, 2, 2, 1.36)
    record_case!(
        payload,
        "hermitian",
        eps_xx,
        eps_yy,
        eps_zz,
        hermitian_offdiagonal(eps_xx, eps_yy, eps_zz)...,
        0.62,
        "periodic",
        "periodic",
        "periodic",
        pml_pair(0, 0),
        pml_pair(0, 0),
        pml_pair(0, 0),
    )

    eps_xx = patterned_epsilon(2, 2, 2, 0.97; imag_scale = 0.2)
    eps_yy = patterned_epsilon(2, 2, 2, 1.13; imag_scale = 0.3)
    eps_zz = patterned_epsilon(2, 2, 2, 1.29; imag_scale = 0.4)
    record_case!(
        payload,
        "lossy",
        eps_xx,
        eps_yy,
        eps_zz,
        lossy_offdiagonal(eps_xx, eps_yy, eps_zz)...,
        0.58,
        "periodic",
        "periodic",
        "periodic",
        pml_pair(0, 0),
        pml_pair(0, 0),
        pml_pair(0, 0),
    )

    eps_xx = patterned_epsilon(3, 3, 3, 1.02)
    eps_yy = patterned_epsilon(3, 3, 3, 1.20)
    eps_zz = patterned_epsilon(3, 3, 3, 1.38)
    record_case!(
        payload,
        "pml",
        eps_xx,
        eps_yy,
        eps_zz,
        hermitian_offdiagonal(eps_xx, eps_yy, eps_zz; scale = 0.7)...,
        0.69,
        "periodic",
        "periodic",
        "periodic",
        pml_pair(1, 1),
        pml_pair(1, 1),
        pml_pair(1, 1),
    )

    matwrite(OUT_PATH, payload)
    println("Wrote ", OUT_PATH)
end

main()
