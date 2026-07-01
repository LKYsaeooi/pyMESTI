"""Generate focused Julia parity fixtures for Step 4 of the Python port.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_step4_julia_fixtures.jl'

The systems here are intentionally tiny. They lock down Julia conventions for
less common 2D TM paths without requiring Julia at Python test time.
"""

using LinearAlgebra
using MAT
using MESTI

const FIXTURE_DIR = @__DIR__
const BLOCH_PATH = joinpath(FIXTURE_DIR, "mesti2s_2d_tm_step4_bloch_continuous.mat")
const NONPERIODIC_PATH = joinpath(FIXTURE_DIR, "mesti2s_2d_tm_step4_nonperiodic.mat")
const SPACER_PATH = joinpath(FIXTURE_DIR, "mesti2s_2d_tm_step4_spacer_wavefront.mat")
const INTERFACE_PATH = joinpath(FIXTURE_DIR, "mesti2s_2d_tm_step4_interface_rt.mat")
const DIRECT_MESTI_PATH = joinpath(FIXTURE_DIR, "mesti_step4_direct_2d_tm.mat")

function base_payload(generator_name)
    return Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => generator_name,
        "julia_mesti_version" => "0.5.1",
        "use_single_precision_MUMPS" => false,
    )
end

function system_payload(syst, epsilon_xx; include_high=true)
    payload = Dict{String,Any}(
        "epsilon_xx" => epsilon_xx,
        "epsilon_low" => syst.epsilon_low,
        "wavelength" => syst.wavelength,
        "dx" => syst.dx,
        "yBC" => syst.yBC,
    )
    if include_high
        payload["epsilon_high"] = syst.epsilon_high
    end
    if isdefined(syst, :ky_B)
        payload["ky_B"] = syst.ky_B
    end
    if isdefined(syst, :zPML) && !isnothing(syst.zPML)
        payload["zPML_npixels"] = syst.zPML[1].npixels
        payload["zPML_npixels_spacer"] = isdefined(syst.zPML[1], :npixels_spacer) && !isnothing(syst.zPML[1].npixels_spacer) ? syst.zPML[1].npixels_spacer : 0
        if length(syst.zPML) > 1
            payload["zPML_high_npixels"] = syst.zPML[2].npixels
            payload["zPML_high_npixels_spacer"] = isdefined(syst.zPML[2], :npixels_spacer) && !isnothing(syst.zPML[2].npixels_spacer) ? syst.zPML[2].npixels_spacer : 0
        end
    end
    return payload
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

function two_sided_channel_payload(channels)
    payload = Dict{String,Any}(
        "kydx_all" => channels.kydx_all,
    )
    merge!(payload, channel_payload("low", channels.low))
    merge!(payload, channel_payload("high", channels.high))
    return payload
end

function wave_matrix(n, m; seed=1.0)
    out = zeros(ComplexF64, n, m)
    for col in 1:m
        for row in 1:n
            out[row, col] = seed * (0.12 * row - 0.05 * col) + im * seed * (0.07 * row + 0.03 * col)
        end
    end
    return out
end

function write_bloch_fixture()
    k0dx = 1.15
    epsilon_xx = ComplexF64[
        1.18+0.01im  1.23+0.00im  1.19+0.02im
        1.14+0.00im  1.21+0.01im  1.26+0.00im
        1.10+0.02im  1.16+0.00im  1.20+0.01im
        1.13+0.00im  1.18+0.02im  1.24+0.00im
        1.17+0.01im  1.22+0.00im  1.25+0.02im
    ]

    syst = Syst()
    syst.epsilon_xx = epsilon_xx
    syst.epsilon_low = 2.25
    syst.epsilon_high = 1.96
    syst.wavelength = 2 * pi / k0dx
    syst.dx = 1.0
    syst.yBC = "Bloch"
    syst.ky_B = 0.21
    syst.zPML = [PML(4)]

    input = channel_type()
    input.side = "low"
    output = channel_type()
    output.side = "high"
    opts = Opts()
    opts.verbal = false
    opts.use_single_precision_MUMPS = false
    opts.use_continuous_dispersion = true
    opts.m0 = 0.35

    S, channels, info = mesti2s(syst, input, output, opts)

    payload = base_payload("Simulation/python/tests/fixtures/generate_mesti2s_step4_julia_fixtures.jl")
    merge!(payload, system_payload(syst, epsilon_xx))
    merge!(payload, two_sided_channel_payload(channels))
    merge!(payload, Dict(
        "description" => "Step 4 2D TM Bloch/ky_B fixture using continuous dispersion and nonzero m0.",
        "use_continuous_dispersion" => true,
        "m0" => opts.m0,
        "S" => S,
        "singular_values" => svd(S).S,
        "return_field_profile" => info.opts.return_field_profile,
    ))
    matwrite(BLOCH_PATH, payload)
    println("wrote ", BLOCH_PATH)
end

function write_nonperiodic_fixture()
    k0dx = 1.35
    epsilon_xx = ComplexF64[
        1.04+0.00im  1.08+0.01im
        1.02+0.01im  1.06+0.00im
        1.07+0.00im  1.10+0.02im
        1.03+0.02im  1.09+0.00im
    ]

    syst = Syst()
    syst.epsilon_xx = epsilon_xx
    syst.epsilon_low = 1.69
    syst.epsilon_high = 1.44
    syst.wavelength = 2 * pi / k0dx
    syst.dx = 1.0
    syst.yBC = "PMC"
    syst.zPML = [PML(4)]

    input = channel_type()
    input.side = "low"
    output = channel_type()
    output.side = "high"
    opts = Opts()
    opts.verbal = false
    opts.use_single_precision_MUMPS = false

    S, channels, info = mesti2s(syst, input, output, opts)

    payload = base_payload("Simulation/python/tests/fixtures/generate_mesti2s_step4_julia_fixtures.jl")
    merge!(payload, system_payload(syst, epsilon_xx))
    merge!(payload, two_sided_channel_payload(channels))
    merge!(payload, Dict(
        "description" => "Step 4 2D TM fixture with a non-periodic transverse PMC boundary.",
        "S" => S,
        "singular_values" => svd(S).S,
        "return_field_profile" => info.opts.return_field_profile,
    ))
    matwrite(NONPERIODIC_PATH, payload)
    println("wrote ", NONPERIODIC_PATH)
end

function write_spacer_wavefront_fixture()
    k0dx = 1.40
    epsilon_xx = ComplexF64[
        1.00+0.00im  1.04+0.01im  1.02+0.00im
        1.03+0.01im  1.07+0.00im  1.06+0.02im
        0.98+0.00im  1.01+0.01im  1.04+0.00im
        1.02+0.02im  1.05+0.00im  1.08+0.01im
    ]

    low_pml = PML(3)
    high_pml = PML(2)
    low_pml.npixels_spacer = 1
    high_pml.npixels_spacer = 2

    syst = Syst()
    syst.epsilon_xx = epsilon_xx
    syst.epsilon_low = 1.21
    syst.epsilon_high = 1.44
    syst.wavelength = 2 * pi / k0dx
    syst.dx = 1.0
    syst.yBC = "periodic"
    syst.zPML = [low_pml, high_pml]

    both = channel_type()
    both.side = "both"
    opts_s = Opts()
    opts_s.verbal = false
    opts_s.use_single_precision_MUMPS = false
    S_both, channels, info_s = mesti2s(syst, both, both, opts_s)

    wf = wavefront()
    wf.v_low = wave_matrix(channels.low.N_prop, 2; seed=1.0)
    wf.v_high = wave_matrix(channels.high.N_prop, 1; seed=-0.75)
    opts_field = Opts()
    opts_field.verbal = false
    opts_field.use_single_precision_MUMPS = false
    opts_field.nz_low = 2
    opts_field.nz_high = 3
    field_profile, _, info_field = mesti2s(syst, wf, opts_field)

    payload = base_payload("Simulation/python/tests/fixtures/generate_mesti2s_step4_julia_fixtures.jl")
    merge!(payload, system_payload(syst, epsilon_xx))
    merge!(payload, two_sided_channel_payload(channels))
    merge!(payload, Dict(
        "description" => "Step 4 2D TM fixture for PML spacers, both-side channels, and mixed low/high wavefront inputs.",
        "S_both" => S_both,
        "S_both_singular_values" => svd(S_both).S,
        "v_low" => wf.v_low,
        "v_high" => wf.v_high,
        "field_profile" => field_profile,
        "nz_low" => opts_field.nz_low,
        "nz_high" => opts_field.nz_high,
        "return_field_profile_S" => info_s.opts.return_field_profile,
        "return_field_profile_field" => info_field.opts.return_field_profile,
    ))
    matwrite(SPACER_PATH, payload)
    println("wrote ", SPACER_PATH)
end

function write_interface_fixture()
    resolution = 10
    n1 = 1.47
    n2 = 2.11
    syst = Syst()
    syst.yBC = "periodic"
    syst.wavelength = 1.0
    syst.dx = 1 / resolution
    syst.zPML = [PML(30)]
    syst.epsilon_low = n1^2
    syst.epsilon_high = n2^2
    syst.epsilon_xx = ones(ComplexF64, 1, 0)

    input = channel_type()
    input.side = "low"
    output = channel_type()
    output.side = "both"
    opts = Opts()
    opts.verbal = false
    opts.use_single_precision_MUMPS = false

    S, channels, info = mesti2s(syst, input, output, opts)
    kzdx_1 = channels.low.kzdx_prop[1]
    kzdx_2 = channels.high.kzdx_prop[1]
    r = (exp(-1im * kzdx_1 / 2) * exp(1im * kzdx_2) - exp(1im * kzdx_1 / 2)) /
        (exp(-1im * kzdx_1 / 2) - exp(1im * kzdx_1 / 2) * exp(1im * kzdx_2))
    t = (sqrt(sin(kzdx_2)) / sqrt(sin(kzdx_1))) *
        (exp(1im * kzdx_1 * 3 / 2) - exp(-1im * kzdx_1 / 2)) /
        (exp(1im * kzdx_2 / 2) * exp(1im * kzdx_1) - exp(-1im * kzdx_2 / 2))

    payload = base_payload("Simulation/python/tests/fixtures/generate_mesti2s_step4_julia_fixtures.jl")
    merge!(payload, system_payload(syst, syst.epsilon_xx))
    merge!(payload, two_sided_channel_payload(channels))
    merge!(payload, Dict(
        "description" => "Step 4 1D interface fixture matching Julia interface_t_r_test.jl.",
        "n_low" => n1,
        "n_high" => n2,
        "S" => S,
        "r_analytic" => r,
        "t_analytic" => t,
        "return_field_profile" => info.opts.return_field_profile,
    ))
    matwrite(INTERFACE_PATH, payload)
    println("wrote ", INTERFACE_PATH)
end

function write_direct_mesti_fixture()
    k0dx = 1.05
    epsilon_xx = ComplexF64[
        1.00+0.01im  1.08+0.00im  1.03+0.02im
        0.96+0.00im  1.04+0.01im  1.10+0.00im
        1.02+0.02im  1.06+0.00im  1.09+0.01im
    ]
    ny, nz = size(epsilon_xx)
    n = ny * nz

    B = zeros(ComplexF64, n, 2)
    C = zeros(ComplexF64, 3, n)
    for row in 1:n
        B[row, 1] = 0.10 * row + 0.03im * row
        B[row, 2] = (-0.05 * row) + 0.02im * (n - row + 1)
    end
    for col in 1:n
        C[1, col] = 0.04 * col - 0.02im * col
        C[2, col] = (-0.03 * col) + 0.01im * (n - col + 1)
        C[3, col] = 0.02 * (-1)^col + 0.015im * col
    end

    syst = Syst()
    syst.epsilon_xx = epsilon_xx
    syst.wavelength = 2 * pi / k0dx
    syst.dx = 1.0
    syst.yBC = "PMC"
    syst.zBC = "PECPMC"

    opts = Opts()
    opts.verbal = false
    opts.use_single_precision_MUMPS = false
    B_struct = Source_struct()
    B_struct.pos = [[1, 1, ny, nz]]
    B_struct.data = [B]

    C_struct = Source_struct()
    C_struct.pos = [[1, 1, ny, nz]]
    C_struct.data = [Matrix(transpose(C))]

    Ex, info_field = mesti(syst, [B_struct], opts)
    projection, info_proj = mesti(syst, [B_struct], [C_struct], opts)

    payload = base_payload("Simulation/python/tests/fixtures/generate_mesti2s_step4_julia_fixtures.jl")
    merge!(payload, Dict(
        "description" => "Step 4 direct 2D TM mesti field and projection fixture.",
        "epsilon_xx" => epsilon_xx,
        "wavelength" => syst.wavelength,
        "dx" => syst.dx,
        "yBC" => syst.yBC,
        "zBC" => syst.zBC,
        "B" => B,
        "C" => C,
        "field_profile" => Ex,
        "projection" => projection,
        "return_field_profile_field" => info_field.opts.return_field_profile,
        "return_field_profile_projection" => info_proj.opts.return_field_profile,
    ))
    matwrite(DIRECT_MESTI_PATH, payload)
    println("wrote ", DIRECT_MESTI_PATH)
end

function main()
    write_bloch_fixture()
    write_nonperiodic_fixture()
    write_spacer_wavefront_fixture()
    write_interface_fixture()
    write_direct_mesti_fixture()
end

main()
