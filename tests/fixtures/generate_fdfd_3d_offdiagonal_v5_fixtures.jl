"""Generate v5 Julia parity fixtures for 3D off-diagonal tensor FDFD assembly.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_fdfd_3d_offdiagonal_v5_fixtures.jl'

The cases stay dense-fixture sized while pinning down the six Yee-staggered
off-diagonal tensor couplings from `mesti_build_fdfd_matrix.jl`.
"""

using MAT
using MESTI
using SparseArrays

const OUT_PATH = joinpath(@__DIR__, "fdfd_3d_offdiagonal_v5.mat")

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
    use_UPML,
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
        use_UPML,
    )

    payload["$(prefix)_epsilon_xx"] = epsilon_xx
    payload["$(prefix)_epsilon_yy"] = epsilon_yy
    payload["$(prefix)_epsilon_zz"] = epsilon_zz
    payload["$(prefix)_epsilon_xy"] = epsilon_xy
    payload["$(prefix)_epsilon_xz"] = epsilon_xz
    payload["$(prefix)_epsilon_yx"] = epsilon_yx
    payload["$(prefix)_epsilon_yz"] = epsilon_yz
    payload["$(prefix)_epsilon_zx"] = epsilon_zx
    payload["$(prefix)_epsilon_zy"] = epsilon_zy
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
        "generator" => "generate_fdfd_3d_offdiagonal_v5_fixtures.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "V5 off-diagonal 3D FDFD tensor parity bundle.",
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
        true,
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
        true,
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
        true,
    )

    eps_xx = patterned_epsilon(2, 3, 2, 0.93)
    eps_yy = patterned_epsilon(3, 3, 2, 1.09)
    eps_zz = patterned_epsilon(3, 3, 2, 1.25)
    record_case!(
        payload,
        "mixed_bc",
        eps_xx,
        eps_yy,
        eps_zz,
        hermitian_offdiagonal(eps_xx, eps_yy, eps_zz; scale = 0.8)...,
        0.64,
        "PMC",
        "PECPMC",
        "PMCPEC",
        pml_pair(0, 0),
        pml_pair(0, 0),
        pml_pair(0, 0),
        true,
    )

    matwrite(OUT_PATH, payload)
    println("Wrote ", OUT_PATH)
end

main()
