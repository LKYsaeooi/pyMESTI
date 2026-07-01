"""Generate the cropped-real Julia double-MUMPS parity fixture.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_ws30_cropped_julia_fixture.jl'

This fixture depends on the external small real-data input documented in the
v2 port plan. It stores the cropped permittivity and reference outputs so the
Python test does not need the external data file at runtime.
"""

using LinearAlgebra
using MAT
using MESTI

const FIXTURE_DIR = @__DIR__
const REAL_ROOT = "/mnt/g/Data/Q_Project/Simulation/tilt/Ws30 Ls7.5/Structure 1"
const INPUT_PATH = joinpath(REAL_ROOT, "epsilon.mat")
const OUTPUT_PATH = joinpath(FIXTURE_DIR, "mesti2s_2d_tm_ws30_center384_double_mumps.mat")

function scalar(data, key)
    value = data[key]
    return Float64(real(value[1]))
end

function main()
    data = matread(INPUT_PATH)
    epsilon_full = data["syst_eps"]
    ny, nz = size(epsilon_full)

    y_start = div(ny - 384, 2) + 1
    z_start = div(nz - 120, 2) + 1
    y_range = y_start:(y_start + 383)
    z_range = z_start:(z_start + 119)
    epsilon_crop = ComplexF64.(epsilon_full[y_range, z_range])

    syst = Syst()
    syst.epsilon_xx = epsilon_crop
    syst.epsilon_low = scalar(data, "epsilon_low")
    syst.epsilon_high = scalar(data, "epsilon_high")
    syst.length_unit = "um"
    syst.wavelength = 0.633
    syst.dx = scalar(data, "region_resolution")
    syst.yBC = "periodic"
    syst.zPML = [PML(Int(round(syst.wavelength / syst.dx)))]

    input = channel_type()
    input.side = "low"
    output = channel_type()
    output.side = "high"

    opts_t = Opts()
    opts_t.verbal = false
    opts_t.use_single_precision_MUMPS = false
    t, channels, info_t = mesti2s(syst, input, output, opts_t)

    svd_t = svd(t)
    v_low = zeros(ComplexF64, channels.low.N_prop, 1)
    v_low[:, 1] = svd_t.V[:, 1]

    wf_input = wavefront()
    wf_input.v_low = v_low
    opts_field = Opts()
    opts_field.verbal = false
    opts_field.use_single_precision_MUMPS = false
    field_profile, _, info_field = mesti2s(syst, wf_input, opts_field)

    payload = Dict(
        "fixture_format" => 1,
        "description" => "Centered cropped-real Ws30 Ls7.5 2D TM Julia double-MUMPS parity fixture.",
        "generator" => "Simulation/python/tests/fixtures/generate_mesti2s_ws30_cropped_julia_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "source_input_path" => INPUT_PATH,
        "epsilon_xx" => epsilon_crop,
        "epsilon_low" => syst.epsilon_low,
        "epsilon_high" => syst.epsilon_high,
        "wavelength" => syst.wavelength,
        "dx" => syst.dx,
        "yBC" => syst.yBC,
        "zPML_npixels" => syst.zPML[1].npixels,
        "crop_name" => "center384y_center120z",
        "y_range_zero_based_half_open" => [first(y_range) - 1, last(y_range)],
        "z_range_zero_based_half_open" => [first(z_range) - 1, last(z_range)],
        "use_single_precision_MUMPS" => false,
        "t" => t,
        "singular_values" => svd_t.S,
        "v_low" => v_low,
        "field_profile" => field_profile,
        "low_N_prop" => channels.low.N_prop,
        "high_N_prop" => channels.high.N_prop,
        "low_ind_prop_julia" => channels.low.ind_prop,
        "high_ind_prop_julia" => channels.high.ind_prop,
        "low_ind_prop_zero_based" => channels.low.ind_prop .- 1,
        "high_ind_prop_zero_based" => channels.high.ind_prop .- 1,
        "low_kzdx_prop" => channels.low.kzdx_prop,
        "high_kzdx_prop" => channels.high.kzdx_prop,
        "return_field_profile_t" => info_t.opts.return_field_profile,
        "return_field_profile_field" => info_field.opts.return_field_profile,
    )

    matwrite(OUTPUT_PATH, payload)
    println("wrote ", OUTPUT_PATH)
    println("N_prop_low=", channels.low.N_prop)
    println("N_prop_high=", channels.high.N_prop)
    println("sigma_open=", svd_t.S[1])
end

main()
