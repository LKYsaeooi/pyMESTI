"""Generate a v5 Julia parity fixture for 2D TM high-level APF defaults.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_apf_default_v5_fixture.jl'

The fixture is intentionally tiny.  It captures Julia's high-level ``mesti2s``
default of using MUMPS/APF for projected scattering-matrix solves, plus a
factorize-and-solve reference on the same deterministic system.
"""

using LinearAlgebra
using MAT
using MESTI

const OUT_PATH = joinpath(@__DIR__, "mesti2s_2d_tm_apf_default_v5.mat")

function system_payload(syst)
    return Dict{String,Any}(
        "epsilon_xx" => syst.epsilon_xx,
        "epsilon_low" => syst.epsilon_low,
        "epsilon_high" => syst.epsilon_high,
        "wavelength" => syst.wavelength,
        "dx" => syst.dx,
        "yBC" => syst.yBC,
        "zPML_npixels" => syst.zPML[1].npixels,
    )
end

function channel_payload(prefix, side)
    return Dict{String,Any}(
        "$(prefix)_N_prop" => side.N_prop,
        "$(prefix)_ind_prop_julia" => side.ind_prop,
        "$(prefix)_ind_prop_zero_based" => side.ind_prop .- 1,
        "$(prefix)_kydx_prop" => side.kydx_prop,
        "$(prefix)_kzdx_prop" => side.kzdx_prop,
        "$(prefix)_sqrt_nu_prop" => side.sqrt_nu_prop,
    )
end

function main()
    k0dx = 1.46
    epsilon_xx = ComplexF64[
        1.07+0.00im  1.12+0.01im  1.09+0.00im
        1.03+0.02im  1.08+0.00im  1.13+0.01im
        1.06+0.00im  1.10+0.02im  1.11+0.00im
        1.02+0.01im  1.09+0.00im  1.14+0.02im
        1.05+0.00im  1.11+0.01im  1.15+0.00im
    ]

    syst = Syst()
    syst.epsilon_xx = epsilon_xx
    syst.epsilon_low = 2.25
    syst.epsilon_high = 2.25
    syst.wavelength = 2 * pi / k0dx
    syst.dx = 1.0
    syst.yBC = "periodic"
    syst.zPML = [PML(4)]

    input = channel_type()
    input.side = "low"
    output = channel_type()
    output.side = "high"

    opts_default = Opts()
    opts_default.verbal = false
    opts_default.solver = "MUMPS"
    opts_default.use_single_precision_MUMPS = false
    S_default, channels, info_default = mesti2s(syst, input, output, opts_default)

    opts_fs = Opts()
    opts_fs.verbal = false
    opts_fs.solver = "JULIA"
    opts_fs.method = "factorize_and_solve"
    opts_fs.use_single_precision_MUMPS = false
    S_fs, _, info_fs = mesti2s(syst, input, output, opts_fs)

    if channels.low.N_prop < 3 || channels.high.N_prop < 3
        error("APF default fixture requires at least three propagating channels on both sides")
    end

    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "Simulation/python/tests/fixtures/generate_mesti2s_apf_default_v5_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "V5 2D TM high-level mumpspy/APF default fixture.",
        "use_single_precision_MUMPS" => false,
        "S_default_apf" => S_default,
        "S_factorize_and_solve" => S_fs,
        "S_default_singular_values" => svd(S_default).S,
        "default_info_method" => info_default.opts.method,
        "fs_info_method" => info_fs.opts.method,
    )
    merge!(payload, system_payload(syst))
    merge!(payload, Dict("kydx_all" => channels.kydx_all))
    merge!(payload, channel_payload("low", channels.low))
    merge!(payload, channel_payload("high", channels.high))

    matwrite(OUT_PATH, payload)
    println("wrote ", OUT_PATH)
    println("S_default size=", size(S_default), " method=", info_default.opts.method)
end

main()
