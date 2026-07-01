"""Generate a reduced fixture for the packaged Gaussian-beam reflection example.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_example_reflection_gaussian_beams_v5_fixture.jl'

The full Julia example is plotting-oriented and much larger.  This fixture
keeps the same source/projection construction, ``C = "transpose(B)"``
reciprocity shortcut, homogeneous reference subtraction, all-side PML, and
PML-excluded field-profile return on a small deterministic grid.
"""

using GeometryPrimitives
using LinearAlgebra
using MAT
using MESTI

const OUT_PATH = joinpath(@__DIR__, "example_reflection_gaussian_beams_v5.mat")

function main()
    syst = Syst()
    syst.length_unit = "um"
    syst.wavelength = 1.0
    syst.dx = syst.wavelength / 4

    pml_npixels = 2
    W = 4.0
    L = 3.0
    r_0 = 0.45
    n_bg = 1.0
    n_scat = 1.2
    y_0 = W / 2
    z_0 = L / 2
    yBC = "PEC"
    zBC = "PEC"

    domain = Cuboid([W / 2, L / 2], [W, L])
    object = [Ball([y_0, z_0], r_0)]
    epsilon_xx = mesti_subpixel_smoothing(
        syst.dx,
        domain,
        n_bg^2,
        object,
        [n_scat^2],
        yBC,
        zBC,
    )

    ny_Ex, nz_Ex = size(epsilon_xx)
    y = collect(syst.dx:syst.dx:(W - syst.dx / 2))
    z = collect(syst.dx:syst.dx:(L - syst.dx / 2))

    NA = 0.5
    z_f = z_0
    y_f = collect(range(0.35 * W, stop=0.65 * W, length=3))
    n_source = pml_npixels + 1
    z_s = z[n_source]
    z_d = z_s
    w_0 = syst.wavelength / (pi * NA)

    E_yf = exp.(-(y .- transpose(y_f)).^2 / (w_0^2))
    channels = mesti_build_channels(ny_Ex, "PEC", (2 * pi / syst.wavelength) * syst.dx, n_bg^2)
    f_transverse = channels.f_x_m(channels.kydx_prop)
    sqrt_nu_prop = reshape(channels.sqrt_nu_prop, :, 1)
    v_f = (sqrt_nu_prop .* adjoint(f_transverse)) * E_yf
    kz = reshape(channels.kzdx_prop / syst.dx, :, 1)
    v_s = exp.(1im * kz * (z_s - z_f)) .* v_f
    B_low = (f_transverse .* transpose(channels.sqrt_nu_prop)) * v_s

    Psi_yf = E_yf
    v_f_tilde = (sqrt_nu_prop .* adjoint(f_transverse)) * Psi_yf
    v_d = exp.(1im * (-kz) * (z_d - z_f)) .* v_f_tilde
    C_low = adjoint(v_d) * (sqrt_nu_prop .* adjoint(f_transverse))
    transpose_mismatch = maximum(abs.(C_low - transpose(B_low)))

    Bx = Source_struct()
    Bx.pos = [[1, n_source, ny_Ex, 1]]
    Bx.data = [B_low]

    pml = PML(pml_npixels)
    pml.direction = "all"
    syst.PML = [pml]
    syst.yBC = yBC
    syst.zBC = zBC

    opts_projection = Opts()
    opts_projection.prefactor = -2im
    opts_projection.verbal = false
    opts_projection.use_single_precision_MUMPS = false

    syst.epsilon_xx = n_bg^2 * ones(ComplexF64, ny_Ex, n_source + pml_npixels)
    t0 = time()
    D, info_reference = mesti(syst, [Bx], "transpose(B)", opts_projection)
    elapsed_reference = time() - t0

    syst.epsilon_xx = epsilon_xx
    t0 = time()
    r, info_reflection = mesti(syst, [Bx], "transpose(B)", D, opts_projection)
    elapsed_reflection = time() - t0

    opts_field = Opts()
    opts_field.prefactor = -2im
    opts_field.verbal = false
    opts_field.use_single_precision_MUMPS = false
    opts_field.exclude_PML_in_field_profiles = true

    t0 = time()
    field_profiles, info_field = mesti(syst, [Bx], opts_field)
    elapsed_field = time() - t0

    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_example_reflection_gaussian_beams_v5_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "Reduced packaged example fixture for 2D Gaussian-beam reflection matrix.",
        "use_single_precision_MUMPS" => false,
        "W" => W,
        "L" => L,
        "r_0" => r_0,
        "n_bg" => n_bg,
        "n_scat" => n_scat,
        "epsilon_bg" => n_bg^2,
        "epsilon_scat" => n_scat^2,
        "y_0" => y_0,
        "z_0" => z_0,
        "wavelength" => syst.wavelength,
        "dx" => syst.dx,
        "pml_npixels" => pml_npixels,
        "yBC" => yBC,
        "zBC" => zBC,
        "epsilon_xx" => epsilon_xx,
        "ny_Ex" => ny_Ex,
        "nz_Ex" => nz_Ex,
        "y" => y,
        "z" => z,
        "NA" => NA,
        "z_f" => z_f,
        "y_f" => y_f,
        "n_source_julia" => n_source,
        "source_plane_index_zero_based" => n_source - 1,
        "z_s" => z_s,
        "w_0" => w_0,
        "N_prop" => channels.N_prop,
        "kydx_prop" => channels.kydx_prop,
        "kzdx_prop" => channels.kzdx_prop,
        "sqrt_nu_prop" => channels.sqrt_nu_prop,
        "B_low" => B_low,
        "C_low" => C_low,
        "C_transpose_B_max_abs_difference" => transpose_mismatch,
        "reference_D" => D,
        "reflection" => r,
        "reflection_abs_squared" => abs2.(r),
        "reflection_singular_values" => svd(r).S,
        "field_profiles" => field_profiles,
        "field_profile_shape" => collect(size(field_profiles)),
        "return_field_profile_reference" => info_reference.opts.return_field_profile,
        "return_field_profile_reflection" => info_reflection.opts.return_field_profile,
        "return_field_profile_field" => info_field.opts.return_field_profile,
        "elapsed_reference_seconds" => elapsed_reference,
        "elapsed_reflection_seconds" => elapsed_reflection,
        "elapsed_field_seconds" => elapsed_field,
    )

    matwrite(OUT_PATH, payload)
    println("Recorded reduced Gaussian-beam reflection example with epsilon size ", size(epsilon_xx))
    println("max(|C_low - transpose(B_low)|) = ", transpose_mismatch)
    println("reflection singular values = ", svd(r).S)
    println("field profile size = ", size(field_profiles))
    println("Wrote ", OUT_PATH)
end

main()
