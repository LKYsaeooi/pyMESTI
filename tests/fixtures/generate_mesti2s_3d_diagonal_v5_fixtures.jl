"""Generate v5 Julia parity fixtures for diagonal 3D ``mesti2s`` boundaries.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_3d_diagonal_v5_fixtures.jl'

The bundle exercises already implemented two-sided diagonal 3D ``mesti2s``
branches that were not covered by the v4 periodic/no-PML fixture: nonzero
z-PML, numeric Bloch phases in x/y, and non-periodic transverse boundaries.
"""

using LinearAlgebra
using MAT
using MESTI

const OUT_PATH = joinpath(@__DIR__, "mesti2s_3d_diagonal_v5_boundaries.mat")

function patterned_epsilon(nx, ny, nz, base)
    epsilon = zeros(ComplexF64, nx, ny, nz)
    for iz in 1:nz
        for iy in 1:ny
            for ix in 1:nx
                epsilon[ix, iy, iz] =
                    base +
                    0.027 * ix +
                    0.015 * iy +
                    0.008 * iz +
                    im * (0.002 * ix - 0.001 * iy + 0.0015 * iz)
            end
        end
    end
    return epsilon
end

function base_system(epsilon_xx, epsilon_yy, epsilon_zz, xBC, yBC, zPML_npixels)
    syst = Syst()
    syst.epsilon_xx = epsilon_xx
    syst.epsilon_yy = epsilon_yy
    syst.epsilon_zz = epsilon_zz
    syst.epsilon_low = 1.0
    syst.epsilon_high = 1.0
    syst.wavelength = 2 * pi / 1.8
    syst.dx = 1.0
    if isa(xBC, Number)
        syst.xBC = "Bloch"
        syst.kx_B = xBC / (size(epsilon_xx, 1) * syst.dx)
    else
        syst.xBC = xBC
    end
    if isa(yBC, Number)
        syst.yBC = "Bloch"
        syst.ky_B = yBC / (size(epsilon_yy, 2) * syst.dx)
    else
        syst.yBC = yBC
    end
    syst.zPML = [PML(zPML_npixels)]
    return syst
end

function base_opts()
    opts = Opts()
    opts.verbal = false
    opts.solver = "JULIA"
    opts.use_single_precision_MUMPS = false
    return opts
end

function all_channels()
    spec = channel_type()
    spec.side = "both"
    spec.polarization = "both"
    return spec
end

function channel_payload!(payload, prefix, side)
    payload["$(prefix)_N_prop"] = side.N_prop
    payload["$(prefix)_ind_prop_julia"] = side.ind_prop
    payload["$(prefix)_ind_prop_zero_based"] = side.ind_prop .- 1
    payload["$(prefix)_kxdx_prop"] = side.kxdx_prop
    payload["$(prefix)_kydx_prop"] = side.kydx_prop
    payload["$(prefix)_kzdx_prop"] = side.kzdx_prop
    payload["$(prefix)_sqrt_nu_prop"] = side.sqrt_nu_prop
end

function record_case!(payload, prefix, syst)
    both = all_channels()
    S, channels, info = mesti2s(syst, both, both, base_opts())

    payload["$(prefix)_epsilon_xx"] = syst.epsilon_xx
    payload["$(prefix)_epsilon_yy"] = syst.epsilon_yy
    payload["$(prefix)_epsilon_zz"] = syst.epsilon_zz
    payload["$(prefix)_epsilon_low"] = syst.epsilon_low
    payload["$(prefix)_epsilon_high"] = syst.epsilon_high
    payload["$(prefix)_wavelength"] = syst.wavelength
    payload["$(prefix)_dx"] = syst.dx
    payload["$(prefix)_xBC"] =
        isdefined(syst, :kx_B) ? syst.kx_B * size(syst.epsilon_xx, 1) * syst.dx : syst.xBC
    payload["$(prefix)_yBC"] =
        isdefined(syst, :ky_B) ? syst.ky_B * size(syst.epsilon_yy, 2) * syst.dx : syst.yBC
    payload["$(prefix)_zPML_npixels"] = syst.zPML[1].npixels
    payload["$(prefix)_kxdx_all"] = channels.kxdx_all
    payload["$(prefix)_kydx_all"] = channels.kydx_all
    payload["$(prefix)_S_both"] = S
    payload["$(prefix)_S_both_singular_values"] = svd(S).S
    payload["$(prefix)_return_field_profile_S"] = info.opts.return_field_profile
    channel_payload!(payload, "$(prefix)_low", channels.low)
    channel_payload!(payload, "$(prefix)_high", channels.high)
    println(
        "Recorded ",
        prefix,
        " with N_prop=(",
        channels.low.N_prop,
        ", ",
        channels.high.N_prop,
        ") and S size ",
        size(S),
    )
end

function main()
    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_mesti2s_3d_diagonal_v5_fixtures.jl",
        "julia_mesti_version" => "0.5.1",
        "solver" => "JULIA",
        "use_single_precision_MUMPS" => false,
        "description" => "V5 two-sided diagonal 3D mesti2s fixture bundle for z-PML, Bloch, and non-periodic transverse boundaries.",
    )

    record_case!(
        payload,
        "pml",
        base_system(
            patterned_epsilon(3, 3, 1, 1.00),
            patterned_epsilon(3, 3, 1, 1.08),
            patterned_epsilon(3, 3, 2, 1.16),
            "periodic",
            "periodic",
            1,
        ),
    )

    record_case!(
        payload,
        "bloch",
        base_system(
            patterned_epsilon(3, 3, 1, 0.98),
            patterned_epsilon(3, 3, 1, 1.06),
            patterned_epsilon(3, 3, 2, 1.14),
            0.37,
            -0.29,
            0,
        ),
    )

    record_case!(
        payload,
        "mixed_bc",
        base_system(
            patterned_epsilon(2, 3, 1, 1.02),
            patterned_epsilon(3, 3, 1, 1.10),
            patterned_epsilon(3, 3, 2, 1.18),
            "PMC",
            "PECPMC",
            0,
        ),
    )

    matwrite(OUT_PATH, payload)
    println("Wrote ", OUT_PATH)
end

main()
