"""Generate a v5 Julia parity fixture for 2D TM ``mesti2s`` symmetrized-K.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_symmetrized_k_v5_fixture.jl'

The fixture is intentionally tiny.  It selects asymmetric low/high channel
subsets so Julia's ``opts.symmetrize_K`` branch must pad the solve channel list
with conjugate-pair output channels before restoring the requested submatrix.
"""

using LinearAlgebra
using MAT
using MESTI

const OUT_PATH = joinpath(@__DIR__, "mesti2s_2d_tm_symmetrized_k_v5.mat")

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
        "$(prefix)_ind_prop_conj_julia" => side.ind_prop_conj,
        "$(prefix)_ind_prop_conj_zero_based" => side.ind_prop_conj .- 1,
        "$(prefix)_kydx_prop" => side.kydx_prop,
        "$(prefix)_kzdx_prop" => side.kzdx_prop,
        "$(prefix)_sqrt_nu_prop" => side.sqrt_nu_prop,
    )
end

function expansion_payload(prefix, side, ind_in, ind_out)
    expanded = sort(unique(vcat(ind_in, side.ind_prop_conj[ind_out])))
    lookup = Dict(expanded[ii] => ii for ii in 1:length(expanded))
    positions = [lookup[ii] for ii in vcat(ind_in, side.ind_prop_conj[ind_out])]
    n_in = length(ind_in)
    return Dict{String,Any}(
        "$(prefix)_expanded_julia" => expanded,
        "$(prefix)_expanded_zero_based" => expanded .- 1,
        "$(prefix)_input_positions_julia" => positions[1:n_in],
        "$(prefix)_input_positions_zero_based" => positions[1:n_in] .- 1,
        "$(prefix)_output_positions_julia" => positions[(n_in + 1):end],
        "$(prefix)_output_positions_zero_based" => positions[(n_in + 1):end] .- 1,
    )
end

function main()
    k0dx = 1.46
    epsilon_xx = ComplexF64[
        1.06+0.00im  1.11+0.01im  1.08+0.00im
        1.02+0.02im  1.09+0.00im  1.14+0.01im
        1.05+0.00im  1.12+0.02im  1.10+0.00im
        1.01+0.01im  1.07+0.00im  1.13+0.02im
        1.04+0.00im  1.10+0.01im  1.16+0.00im
    ]

    syst = Syst()
    syst.epsilon_xx = epsilon_xx
    syst.epsilon_low = 2.25
    syst.epsilon_high = 2.25
    syst.wavelength = 2 * pi / k0dx
    syst.dx = 1.0
    syst.yBC = "periodic"
    syst.zPML = [PML(4)]

    channels = mesti_build_channels(
        size(epsilon_xx, 1),
        syst.yBC,
        k0dx,
        syst.epsilon_low,
        syst.epsilon_high,
        false,
    )
    if channels.low.N_prop < 3 || channels.high.N_prop < 3
        error("symmetrized-K fixture requires at least three propagating channels on both sides")
    end

    input = channel_index()
    input.ind_low = [2]
    input.ind_high = [1, 3]
    output = channel_index()
    output.ind_low = [1, 3]
    output.ind_high = [2]

    opts_sym = Opts()
    opts_sym.verbal = false
    opts_sym.solver = "MUMPS"
    opts_sym.method = "APF"
    opts_sym.use_single_precision_MUMPS = false
    opts_sym.symmetrize_K = true
    S_sym, channels, info_sym = mesti2s(syst, input, output, opts_sym)

    opts_unsym = Opts()
    opts_unsym.verbal = false
    opts_unsym.solver = "JULIA"
    opts_unsym.method = "factorize_and_solve"
    opts_unsym.use_single_precision_MUMPS = false
    opts_unsym.symmetrize_K = false
    S_unsym, _, info_unsym = mesti2s(syst, input, output, opts_unsym)

    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "Simulation/python/tests/fixtures/generate_mesti2s_symmetrized_k_v5_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "V5 2D TM symmetrized-K channel padding/permutation fixture.",
        "use_single_precision_MUMPS" => false,
        "input_ind_low_julia" => input.ind_low,
        "input_ind_high_julia" => input.ind_high,
        "output_ind_low_julia" => output.ind_low,
        "output_ind_high_julia" => output.ind_high,
        "input_ind_low_zero_based" => input.ind_low .- 1,
        "input_ind_high_zero_based" => input.ind_high .- 1,
        "output_ind_low_zero_based" => output.ind_low .- 1,
        "output_ind_high_zero_based" => output.ind_high .- 1,
        "S_sym" => S_sym,
        "S_unsym" => S_unsym,
        "S_sym_singular_values" => svd(S_sym).S,
        "symmetrize_K_info" => info_sym.opts.symmetrize_K,
        "unsymmetrized_info" => info_unsym.opts.symmetrize_K,
    )
    merge!(payload, system_payload(syst))
    merge!(payload, Dict("kydx_all" => channels.kydx_all))
    merge!(payload, channel_payload("low", channels.low))
    merge!(payload, channel_payload("high", channels.high))
    merge!(payload, expansion_payload("low", channels.low, input.ind_low, output.ind_low))
    merge!(payload, expansion_payload("high", channels.high, input.ind_high, output.ind_high))
    payload["padded_result_shape"] = [
        length(payload["low_expanded_julia"]) + length(payload["high_expanded_julia"]),
        length(payload["low_expanded_julia"]) + length(payload["high_expanded_julia"]),
    ]

    matwrite(OUT_PATH, payload)
    println("wrote ", OUT_PATH)
    println("S_sym size=", size(S_sym), " padded=", payload["padded_result_shape"])
end

main()
