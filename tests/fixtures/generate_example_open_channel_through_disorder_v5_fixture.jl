"""Generate a reduced fixture for the packaged open-channel disorder example.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_example_open_channel_through_disorder_v5_fixture.jl'

The full Julia example uses a large random-cylinder sample and plotting/GIF
output. This fixture keeps the transmission-matrix, SVD open-channel,
wavefront field-profile, and direct ``mesti`` source-comparison calculations
on a small deterministic system suitable for Python/SciPy regression tests.
"""

using GeometryPrimitives
using LinearAlgebra
using MAT
using MESTI

const OUT_PATH = joinpath(@__DIR__, "example_open_channel_through_disorder_v5.mat")
const JULIA_EXAMPLE_DIR = normpath(joinpath(@__DIR__, "..", "..", "..", "julia", "MESTI.jl-0.5.1", "examples", "2d_open_channel_through_disorder"))

include(joinpath(JULIA_EXAMPLE_DIR, "build_epsilon_disorder.jl"))

function main()
    dx = 0.25
    W = 2.0
    L = 1.5
    L_tot = 2.5
    r_min = 0.18
    r_max = 0.26
    min_sep = 0.03
    number_density = 1.0
    rng_seed = 0

    epsilon_scat = 1.2^2
    epsilon_bg = 1.0^2
    epsilon_low = 1.0^2
    epsilon_high = 1.0^2
    yBC = "periodic"

    epsilon_xx, y0_list, z0_list, r0_list, y_Ex, z_Ex = build_epsilon_disorder(
        W,
        L,
        r_min,
        r_max,
        min_sep,
        number_density,
        rng_seed,
        dx,
        epsilon_scat,
        epsilon_bg,
        true,
    )

    pml_npixels = 16
    syst = Syst()
    syst.epsilon_xx = epsilon_xx
    syst.epsilon_low = epsilon_low
    syst.epsilon_high = epsilon_high
    syst.length_unit = "lambda_0"
    syst.wavelength = 1.0
    syst.dx = dx
    syst.yBC = yBC
    syst.zPML = [PML(pml_npixels)]

    input = channel_type()
    input.side = "low"
    output = channel_type()
    output.side = "high"
    opts_transmission = Opts()
    opts_transmission.verbal = false
    opts_transmission.use_single_precision_MUMPS = false

    t0 = time()
    t, channels, info_transmission = mesti2s(syst, input, output, opts_transmission)
    elapsed_transmission = time() - t0

    _, sigma, v = svd(t)
    tau = sigma.^2
    v_open = v[:, 1]
    N_prop_low = channels.low.N_prop
    ind_normal = round(Int, (N_prop_low + 1) / 2)
    ind_normal_zero_based = ind_normal - 1

    T_avg = sum(abs2.(t)) / N_prop_low
    T_PW = sum(abs2.(t[:, ind_normal]))
    T_open = sigma[1]^2

    v_low = zeros(ComplexF64, N_prop_low, 2)
    v_low[ind_normal, 1] = 1
    v_low[:, 2] = v_open
    input_wavefront = wavefront()
    input_wavefront.v_low = v_low

    nz_low = round(Int, (L_tot - L) / 2 / dx)
    nz_high = nz_low
    opts_field = Opts()
    opts_field.nz_low = nz_low
    opts_field.nz_high = nz_high
    opts_field.use_L0_threads = false
    opts_field.verbal = false
    opts_field.use_single_precision_MUMPS = false

    t0 = time()
    Ex, _, info_field = mesti2s(syst, input_wavefront, opts_field)
    elapsed_field = time() - t0

    direct_syst = Syst()
    ny_Ex, nz_Ex = size(epsilon_xx)
    direct_syst.epsilon_xx = cat(
        epsilon_low * ones(ComplexF64, ny_Ex, pml_npixels + 1),
        epsilon_xx,
        epsilon_high * ones(ComplexF64, ny_Ex, pml_npixels + 1),
        dims=2,
    )
    direct_syst.length_unit = "lambda_0"
    direct_syst.wavelength = 1.0
    direct_syst.dx = dx
    direct_syst.yBC = yBC
    pml = PML(pml_npixels)
    pml.direction = "z"
    direct_syst.PML = [pml]

    Bx = Source_struct()
    f_prop_low_Ex = channels.f_x_m(channels.low.kydx_prop)
    source_weights = channels.low.sqrt_nu_prop .* exp.((-1im * 0.5) .* channels.low.kzdx_prop)
    B_Ex_low = f_prop_low_Ex * (source_weights .* v_low)
    Bx.data = [B_Ex_low]
    Bx.pos = [[1, pml_npixels + 1, ny_Ex, 1]]

    opts_direct = Opts()
    opts_direct.prefactor = -2im
    opts_direct.use_L0_threads = false
    opts_direct.verbal = false
    opts_direct.use_single_precision_MUMPS = false

    t0 = time()
    Ex_prime, info_direct = mesti(direct_syst, [Bx], opts_direct)
    elapsed_direct = time() - t0

    mesti2s_core = Ex[:, nz_low+1:end-nz_high-1, :]
    direct_core = Ex_prime[:, pml_npixels+1+1:end-pml_npixels-1-1, :]
    direct_field_difference_max = maximum(abs.(mesti2s_core - direct_core))

    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_example_open_channel_through_disorder_v5_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "Reduced packaged example fixture for 2D open channel through disorder.",
        "use_single_precision_MUMPS" => false,
        "W" => W,
        "L" => L,
        "L_tot" => L_tot,
        "r_min" => r_min,
        "r_max" => r_max,
        "min_sep" => min_sep,
        "number_density" => number_density,
        "rng_seed" => rng_seed,
        "epsilon_scat" => epsilon_scat,
        "epsilon_bg" => epsilon_bg,
        "epsilon_low" => epsilon_low,
        "epsilon_high" => epsilon_high,
        "wavelength" => syst.wavelength,
        "dx" => dx,
        "pml_npixels" => pml_npixels,
        "yBC" => yBC,
        "epsilon_xx" => epsilon_xx,
        "ny_Ex" => ny_Ex,
        "nz_Ex" => nz_Ex,
        "y_Ex" => collect(y_Ex),
        "z_Ex" => collect(z_Ex),
        "y0_list" => y0_list,
        "z0_list" => z0_list,
        "r0_list" => r0_list,
        "N_prop_low" => N_prop_low,
        "N_prop_high" => channels.high.N_prop,
        "low_kydx_prop" => channels.low.kydx_prop,
        "low_kzdx_prop" => channels.low.kzdx_prop,
        "low_sqrt_nu_prop" => channels.low.sqrt_nu_prop,
        "high_kydx_prop" => channels.high.kydx_prop,
        "high_kzdx_prop" => channels.high.kzdx_prop,
        "transmission" => t,
        "singular_values" => sigma,
        "transmission_eigenvalues" => tau,
        "open_channel" => v_open,
        "normal_index_julia" => ind_normal,
        "normal_index_zero_based" => ind_normal_zero_based,
        "v_low" => v_low,
        "T_avg" => T_avg,
        "T_PW" => T_PW,
        "T_open" => T_open,
        "nz_low" => nz_low,
        "nz_high" => nz_high,
        "field_profiles" => Ex,
        "field_profile_shape" => collect(size(Ex)),
        "direct_field_profiles" => Ex_prime,
        "direct_field_profile_shape" => collect(size(Ex_prime)),
        "direct_field_difference_max" => direct_field_difference_max,
        "return_field_profile_transmission" => info_transmission.opts.return_field_profile,
        "return_field_profile_field" => info_field.opts.return_field_profile,
        "return_field_profile_direct" => info_direct.opts.return_field_profile,
        "elapsed_transmission_seconds" => elapsed_transmission,
        "elapsed_field_seconds" => elapsed_field,
        "elapsed_direct_seconds" => elapsed_direct,
    )

    matwrite(OUT_PATH, payload)
    println("Recorded reduced open-channel disorder example with epsilon size ", size(epsilon_xx))
    println("transmission singular values = ", sigma)
    println("T_avg = ", T_avg, ", T_PW = ", T_PW, ", T_open = ", T_open)
    println("field profile size = ", size(Ex))
    println("direct field difference max = ", direct_field_difference_max)
    println("Wrote ", OUT_PATH)
end

main()
