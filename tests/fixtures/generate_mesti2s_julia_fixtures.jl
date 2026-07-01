"""Generate Julia parity fixtures for the Python 2D TM ``mesti2s`` tests.

Run from the project root through the WSL Julia environment, for example:

    wsl --user lky -- bash -lc 'source /opt/intel/oneapi/mpi/2021.13/env/vars.sh >/dev/null 2>&1; export LD_LIBRARY_PATH="/opt/intel/oneapi/2024.2/lib:\${LD_LIBRARY_PATH}"; cd "/mnt/d/BaiduSyncdisk/Projects/Q project"; /home/lky/julia/julia-1.10.5/bin/julia Simulation/python/tests/fixtures/generate_mesti2s_julia_fixtures.jl'

The generated MAT files are committed fixtures. Python unittests load them
directly and do not require Julia at test time.
"""

using LinearAlgebra
using MAT
using MESTI

const FIXTURE_DIR = @__DIR__
const LOW_TO_HIGH_PATH = joinpath(FIXTURE_DIR, "mesti2s_2d_tm_julia_low_to_high.mat")
const WAVEFRONT_PATH = joinpath(FIXTURE_DIR, "mesti2s_2d_tm_julia_wavefront_v_low.mat")

function fixture_system()
    k0dx = 1.3
    epsilon_xx = ComplexF64[
        1.00+0.00im  1.05+0.02im  1.08+0.00im
        1.03+0.01im  1.10+0.00im  1.12+0.03im
        0.98+0.00im  1.02+0.01im  1.07+0.02im
        1.04+0.02im  1.01+0.00im  1.09+0.01im
    ]
    v_low = ComplexF64[
        1.0+0.0im    0.25-0.5im
        0.0+0.5im   -0.75+0.25im
       -0.25+0.75im  0.5+0.0im
    ]

    syst = Syst()
    syst.epsilon_xx = epsilon_xx
    syst.epsilon_low = 1.21
    syst.epsilon_high = 1.44
    syst.wavelength = 2 * pi / k0dx
    syst.dx = 1.0
    syst.yBC = "periodic"
    syst.zPML = [PML(3)]
    return syst, epsilon_xx, v_low
end

function system_payload(syst, epsilon_xx)
    return Dict(
        "fixture_format" => 1,
        "generator" => "Simulation/python/tests/fixtures/generate_mesti2s_julia_fixtures.jl",
        "julia_mesti_version" => "0.5.1",
        "epsilon_xx" => epsilon_xx,
        "epsilon_low" => syst.epsilon_low,
        "epsilon_high" => syst.epsilon_high,
        "wavelength" => syst.wavelength,
        "dx" => syst.dx,
        "yBC" => syst.yBC,
        "zPML_npixels" => syst.zPML[1].npixels,
    )
end

function main()
    syst, epsilon_xx, v_low = fixture_system()

    input = channel_type()
    input.side = "low"
    output = channel_type()
    output.side = "high"
    opts = Opts()
    opts.verbal = false

    t, channels_t, info_t = mesti2s(syst, input, output, opts)
    singular_values = svd(t).S

    t_payload = system_payload(syst, epsilon_xx)
    merge!(t_payload, Dict(
        "description" => "Julia-generated 2D TM mesti2s low-to-high transmission fixture.",
        "t" => t,
        "singular_values" => singular_values,
        "low_N_prop" => channels_t.low.N_prop,
        "high_N_prop" => channels_t.high.N_prop,
        "low_ind_prop_julia" => channels_t.low.ind_prop,
        "high_ind_prop_julia" => channels_t.high.ind_prop,
        "low_ind_prop_zero_based" => channels_t.low.ind_prop .- 1,
        "high_ind_prop_zero_based" => channels_t.high.ind_prop .- 1,
        "low_kzdx_prop" => channels_t.low.kzdx_prop,
        "high_kzdx_prop" => channels_t.high.kzdx_prop,
        "return_field_profile" => info_t.opts.return_field_profile,
    ))
    matwrite(LOW_TO_HIGH_PATH, t_payload)

    wf_input = wavefront()
    wf_input.v_low = v_low
    field_profile, channels_field, info_field = mesti2s(syst, wf_input, opts)

    field_payload = system_payload(syst, epsilon_xx)
    merge!(field_payload, Dict(
        "description" => "Julia-generated 2D TM mesti2s wavefront.v_low field-profile fixture.",
        "v_low" => v_low,
        "field_profile" => field_profile,
        "low_N_prop" => channels_field.low.N_prop,
        "high_N_prop" => channels_field.high.N_prop,
        "low_ind_prop_julia" => channels_field.low.ind_prop,
        "high_ind_prop_julia" => channels_field.high.ind_prop,
        "low_ind_prop_zero_based" => channels_field.low.ind_prop .- 1,
        "high_ind_prop_zero_based" => channels_field.high.ind_prop .- 1,
        "return_field_profile" => info_field.opts.return_field_profile,
    ))
    matwrite(WAVEFRONT_PATH, field_payload)

    println("wrote ", LOW_TO_HIGH_PATH)
    println("wrote ", WAVEFRONT_PATH)
end

main()
