"""Generate the v3 Julia parity fixture for direct 3D diagonal ``mesti``.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti_3d_direct_julia_fixture.jl'

The system is deliberately tiny and has no PML.  It exercises the high-level
direct solve layer above the already-fixture-backed 3D diagonal FDFD matrix:
dense RHS field profiles, dense projected solves with ``D``, and Julia
``Source_struct`` assembly for component-wise sources and projections.
"""

using MAT
using MESTI

const OUT_PATH = joinpath(@__DIR__, "mesti_3d_direct_diagonal_pec.mat")

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

function component_rhs(nx, ny, nz, nrhs, scale)
    out = zeros(ComplexF64, nx * ny * nz, nrhs)
    for a in 1:nrhs
        for iz in 1:nz
            for iy in 1:ny
                for ix in 1:nx
                    row = ix + (iy - 1) * nx + (iz - 1) * nx * ny
                    out[row, a] =
                        scale * (0.031 * ix - 0.017 * iy + 0.011 * iz + 0.043 * a) +
                        im * scale * (0.019 * ix + 0.023 * iy - 0.013 * iz + 0.029 * a)
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
                0.007 * row + 0.003 * col +
                im * (0.005 * row - 0.002 * col)
        end
    end
    return out
end

function direct_matrix(noutputs, nrhs)
    out = zeros(ComplexF64, noutputs, nrhs)
    for row in 1:noutputs
        for col in 1:nrhs
            out[row, col] = 0.013 * row - 0.004 * col + im * (0.002 * row + 0.006 * col)
        end
    end
    return out
end

function source_struct_from_component(data, nx, ny, nz)
    source = Source_struct()
    source.pos = [[1, 1, 1, nx, ny, nz]]
    source.data = [reshape(data, nx, ny, nz, size(data, 2))]
    return source
end

function projection_struct_from_component(data, nx, ny, nz)
    projection = Source_struct()
    projection.pos = [[1, 1, 1, nx, ny, nz]]
    projection.data = [reshape(Matrix(transpose(data)), nx, ny, nz, size(data, 1))]
    return projection
end

function main()
    epsilon_xx = patterned_epsilon(3, 2, 2, 1.0)
    epsilon_yy = patterned_epsilon(2, 3, 2, 1.5)
    epsilon_zz = patterned_epsilon(2, 2, 3, 2.0)

    nt_Ex = length(epsilon_xx)
    nt_Ey = length(epsilon_yy)
    nt_Ez = length(epsilon_zz)
    ntotal = nt_Ex + nt_Ey + nt_Ez
    nrhs = 2
    noutputs = 4

    Bx = component_rhs(size(epsilon_xx)..., nrhs, 1.0)
    By = component_rhs(size(epsilon_yy)..., nrhs, -0.7)
    Bz = component_rhs(size(epsilon_zz)..., nrhs, 1.3)
    B = [Bx; By; Bz]

    C = projection_matrix(noutputs, ntotal)
    Cx = C[:, 1:nt_Ex]
    Cy = C[:, (nt_Ex + 1):(nt_Ex + nt_Ey)]
    Cz = C[:, (nt_Ex + nt_Ey + 1):ntotal]
    D = direct_matrix(noutputs, nrhs)

    syst = Syst()
    syst.epsilon_xx = epsilon_xx
    syst.epsilon_yy = epsilon_yy
    syst.epsilon_zz = epsilon_zz
    syst.wavelength = 2 * pi / 0.73
    syst.dx = 1.0
    syst.xBC = "PEC"
    syst.yBC = "PEC"
    syst.zBC = "PEC"

    opts = Opts()
    opts.verbal = false
    opts.use_single_precision_MUMPS = false

    k0dx = (2 * pi / syst.wavelength) * syst.dx
    zero_pml = [PML(0), PML(0)]
    A, is_symmetric_A, _, _, _ = mesti_build_fdfd_matrix(
        epsilon_xx,
        epsilon_yy,
        epsilon_zz,
        k0dx,
        syst.xBC,
        syst.yBC,
        syst.zBC,
        zero_pml,
        zero_pml,
        zero_pml,
        true,
    )
    X = A \ B
    Ex = reshape(X[1:nt_Ex, :], size(epsilon_xx)..., nrhs)
    Ey = reshape(X[(nt_Ex + 1):(nt_Ex + nt_Ey), :], size(epsilon_yy)..., nrhs)
    Ez = reshape(X[(nt_Ex + nt_Ey + 1):ntotal, :], size(epsilon_zz)..., nrhs)
    projection_with_D = C * X - D

    B_struct = [
        source_struct_from_component(Bx, size(epsilon_xx)...),
        source_struct_from_component(By, size(epsilon_yy)...),
        source_struct_from_component(Bz, size(epsilon_zz)...),
    ]
    C_struct = [
        projection_struct_from_component(Cx, size(epsilon_xx)...),
        projection_struct_from_component(Cy, size(epsilon_yy)...),
        projection_struct_from_component(Cz, size(epsilon_zz)...),
    ]
    Ex_struct, Ey_struct, Ez_struct, info_struct_field = mesti(syst, B_struct, opts)
    projection_struct_with_D, info_struct_projection = mesti(syst, B_struct, C_struct, D, opts)

    matwrite(
        OUT_PATH,
        Dict{String,Any}(
            "fixture_format" => 1,
            "generator" => "generate_mesti_3d_direct_julia_fixture.jl",
            "julia_mesti_version" => "0.5.1",
            "use_single_precision_MUMPS" => false,
            "epsilon_xx" => epsilon_xx,
            "epsilon_yy" => epsilon_yy,
            "epsilon_zz" => epsilon_zz,
            "wavelength" => syst.wavelength,
            "dx" => syst.dx,
            "xBC" => syst.xBC,
            "yBC" => syst.yBC,
            "zBC" => syst.zBC,
            "B" => B,
            "C" => C,
            "D" => D,
            "is_symmetric_A" => is_symmetric_A,
            "field_Ex" => Ex,
            "field_Ey" => Ey,
            "field_Ez" => Ez,
            "field_struct_Ex" => Ex_struct,
            "field_struct_Ey" => Ey_struct,
            "field_struct_Ez" => Ez_struct,
            "projection_with_D" => projection_with_D,
            "projection_struct_with_D" => projection_struct_with_D,
            "return_field_profile_field" => true,
            "return_field_profile_projection" => false,
            "return_field_profile_struct_field" => info_struct_field.opts.return_field_profile,
            "return_field_profile_struct_projection" => info_struct_projection.opts.return_field_profile,
        ),
    )
    println("Wrote ", OUT_PATH, " with ", ntotal, " unknowns, ", nrhs, " RHS, and ", noutputs, " projections")
end

main()
