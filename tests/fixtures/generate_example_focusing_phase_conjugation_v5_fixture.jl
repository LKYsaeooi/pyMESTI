"""Generate a reduced fixture for the packaged phase-conjugation focusing example.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_example_focusing_phase_conjugation_v5_fixture.jl'

The full Julia example projects a point source inside a large random-cylinder
sample onto low-side propagating channels, phase-conjugates those coefficients,
and compares regular versus phase-conjugated focusing fields. This fixture
keeps that numerical core on a tiny deterministic system and omits plotting
and GIF output.
"""

using GeometryPrimitives
using LinearAlgebra
using MAT
using MESTI
using Statistics

const OUT_PATH = joinpath(@__DIR__, "example_focusing_phase_conjugation_v5.mat")
const JULIA_EXAMPLE_DIR = normpath(joinpath(@__DIR__, "..", "..", "..", "julia", "MESTI.jl-0.5.1", "examples", "2d_focusing_inside_disorder_with_phase_conjugation"))

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
        true;
        no_scatterer_center = true,
    )

    pml_npixels = 16
    ny_Ex, nz_Ex = size(epsilon_xx)
    m0_focus = Int((W / dx) / 2)
    l0_focus = Int((L / dx) / 2)
    m0_focus_zero_based = m0_focus - 1
    l0_focus_zero_based = l0_focus - 1

    syst_projection = Syst()
    syst_projection.length_unit = "lambda_0"
    syst_projection.wavelength = 1.0
    syst_projection.dx = dx
    syst_projection.yBC = yBC
    syst_projection.epsilon_xx = cat(
        epsilon_low * ones(ComplexF64, ny_Ex, pml_npixels + 1),
        epsilon_xx,
        epsilon_high * ones(ComplexF64, ny_Ex, pml_npixels + 1),
        dims=2,
    )

    Bx = Source_struct()
    Bx.pos = [[m0_focus, l0_focus + pml_npixels + 1, 1, 1]]
    Bx.data = [ones(ComplexF64, 1, 1)]

    pml = PML(pml_npixels)
    pml.direction = "z"
    syst_projection.PML = [pml]

    k0dx = 2 * pi * syst_projection.wavelength * dx
    channels_low = mesti_build_channels(ny_Ex, yBC, k0dx, epsilon_low)
    N_prop_low = channels_low.N_prop

    Cx = Source_struct()
    Cx.pos = [[1, pml_npixels + 1, ny_Ex, 1]]
    C_low = (
        conj.(channels_low.f_x_m(channels_low.kydx_prop)) .*
        reshape(channels_low.sqrt_nu_prop, 1, :) .*
        reshape(exp.((-1im * 0.5) .* channels_low.kzdx_prop), 1, :)
    )
    Cx.data = [C_low]

    opts_projection = Opts()
    opts_projection.verbal = false
    opts_projection.use_L0_threads = false
    opts_projection.use_single_precision_MUMPS = false

    t0 = time()
    w, info_projection = mesti(syst_projection, [Bx], [Cx], opts_projection)
    elapsed_projection = time() - t0

    opts_point_field = Opts()
    opts_point_field.verbal = false
    opts_point_field.use_L0_threads = false
    opts_point_field.use_single_precision_MUMPS = false

    t0 = time()
    Ex_point, info_point_field = mesti(syst_projection, [Bx], opts_point_field)
    elapsed_point_field = time() - t0
    point_slice = reshape(Ex_point[:, pml_npixels + 1, :], ny_Ex, :)
    w_from_field = transpose(C_low) * point_slice
    projection_from_field_difference_max = maximum(abs.(w - w_from_field))

    syst = Syst()
    syst.epsilon_xx = epsilon_xx
    syst.length_unit = "lambda_0"
    syst.wavelength = 1.0
    syst.dx = dx
    syst.yBC = yBC
    syst.epsilon_low = epsilon_low
    syst.epsilon_high = epsilon_high
    syst.zPML = [PML(pml_npixels)]

    epsilon_ave = mean(epsilon_xx)
    channels_ave_epsilon = mesti_build_channels(ny_Ex, yBC, k0dx, epsilon_ave)
    N_prop_ave_epsilon = channels_ave_epsilon.N_prop

    wf_reg_focus_full = (
        exp.((-1im) .* channels_ave_epsilon.kydx_prop .* m0_focus) .*
        exp.((-1im) .* channels_ave_epsilon.kzdx_prop .* (l0_focus - 0.5))
    )
    channel_diff = N_prop_ave_epsilon - N_prop_low
    if channel_diff < 0 || channel_diff % 2 != 0
        error("Reduced fixture expected average-epsilon propagating channels to center-crop onto low-side channels.")
    end
    crop_each_side = div(channel_diff, 2)
    wf_reg_focus = wf_reg_focus_full[crop_each_side + 1:end - crop_each_side]
    wf_reg_focus = wf_reg_focus / norm(wf_reg_focus)

    phase_conjugated_focus = conj(vec(w))[channels_low.ind_prop_conj] / norm(w)

    v_low = zeros(ComplexF64, N_prop_low, 2)
    v_low[:, 1] = wf_reg_focus
    v_low[:, 2] = phase_conjugated_focus
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
    Ex, channels_field, info_field = mesti2s(syst, input_wavefront, opts_field)
    elapsed_field = time() - t0

    normalization_factor = maximum(abs.(Ex[:, :, 2]))
    Ex_normalized = Ex / normalization_factor
    focus_z_extended = nz_low + l0_focus
    focus_z_extended_zero_based = focus_z_extended - 1
    focus_intensities = abs2.(vec(Ex[m0_focus, focus_z_extended, :]))
    focus_intensity_ratio = focus_intensities[2] / focus_intensities[1]

    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_example_focusing_phase_conjugation_v5_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "Reduced packaged example fixture for 2D focusing inside disorder with phase conjugation.",
        "use_single_precision_MUMPS" => false,
        "W" => W,
        "L" => L,
        "L_tot" => L_tot,
        "r_min" => r_min,
        "r_max" => r_max,
        "min_sep" => min_sep,
        "number_density" => number_density,
        "rng_seed" => rng_seed,
        "no_scatterer_center" => true,
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
        "m0_focus_julia" => m0_focus,
        "l0_focus_julia" => l0_focus,
        "focus_y_index_zero_based" => m0_focus_zero_based,
        "focus_z_index_zero_based" => l0_focus_zero_based,
        "focus_z_extended_julia" => focus_z_extended,
        "focus_z_extended_zero_based" => focus_z_extended_zero_based,
        "N_prop_low" => N_prop_low,
        "N_prop_ave_epsilon" => N_prop_ave_epsilon,
        "channel_crop_each_side" => crop_each_side,
        "low_kydx_prop" => channels_low.kydx_prop,
        "low_kzdx_prop" => channels_low.kzdx_prop,
        "low_sqrt_nu_prop" => channels_low.sqrt_nu_prop,
        "low_ind_prop_conj_julia" => channels_low.ind_prop_conj,
        "low_ind_prop_conj_zero_based" => channels_low.ind_prop_conj .- 1,
        "ave_kydx_prop" => channels_ave_epsilon.kydx_prop,
        "ave_kzdx_prop" => channels_ave_epsilon.kzdx_prop,
        "projection_C_low" => C_low,
        "projected_coefficients" => w,
        "projection_from_field" => w_from_field,
        "projection_from_field_difference_max" => projection_from_field_difference_max,
        "regular_focus_wavefront_full" => wf_reg_focus_full,
        "regular_focus_wavefront" => wf_reg_focus,
        "phase_conjugated_wavefront" => phase_conjugated_focus,
        "v_low" => v_low,
        "nz_low" => nz_low,
        "nz_high" => nz_high,
        "field_profiles" => Ex,
        "normalized_field_profiles" => Ex_normalized,
        "normalization_factor" => normalization_factor,
        "field_profile_shape" => collect(size(Ex)),
        "focus_intensities" => focus_intensities,
        "regular_focus_intensity" => focus_intensities[1],
        "phase_conjugated_focus_intensity" => focus_intensities[2],
        "phase_to_regular_intensity_ratio" => focus_intensity_ratio,
        "return_field_profile_projection" => info_projection.opts.return_field_profile,
        "return_field_profile_point_source" => info_point_field.opts.return_field_profile,
        "return_field_profile_field" => info_field.opts.return_field_profile,
        "elapsed_projection_seconds" => elapsed_projection,
        "elapsed_point_field_seconds" => elapsed_point_field,
        "elapsed_field_seconds" => elapsed_field,
    )

    matwrite(OUT_PATH, payload)
    println("Recorded reduced phase-conjugation focusing example with epsilon size ", size(epsilon_xx))
    println("projected coefficient norm = ", norm(w))
    println("projection-from-field max difference = ", projection_from_field_difference_max)
    println("field profile size = ", size(Ex))
    println("focus intensities = ", focus_intensities, ", ratio = ", focus_intensity_ratio)
    println("Wrote ", OUT_PATH)
end

main()
