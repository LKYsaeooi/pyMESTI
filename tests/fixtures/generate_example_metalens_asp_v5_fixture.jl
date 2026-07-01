"""Generate a reduced fixture for the packaged metalens ASP example.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_example_metalens_asp_v5_fixture.jl'

The full Julia example loads a production-size wide-FOV metalens and then
propagates sampled transmitted fields to the focal plane with angular spectrum
propagation. This fixture keeps the source/projection, direct ``mesti`` solve,
and ASP numerical core on a tiny deterministic lens and omits plotting,
animation, and the large design file.
"""

using FFTW
using LinearAlgebra
using MAT
using MESTI

const OUT_PATH = joinpath(@__DIR__, "example_metalens_asp_v5.mat")
const JULIA_EXAMPLE_DIR = normpath(joinpath(@__DIR__, "..", "..", "..", "julia", "MESTI.jl-0.5.1", "examples", "2d_metalens_focusing_via_angular_spectrum_propagation"))

include(joinpath(JULIA_EXAMPLE_DIR, "asp.jl"))

function demo_epsilon_metalens(ny::Int, nz::Int, n_struct::Float64)
    epsilon = ones(ComplexF64, ny, nz)
    center = (ny + 1) / 2
    for m in 1:ny
        normalized_radius = abs(m - center) / (ny / 2)
        filled_cols = clamp(round(Int, nz * (0.28 + 0.52 * (1 - normalized_radius))), 1, nz)
        parity_offset = isodd(m) ? 0 : 1
        for n in 1:nz
            if n <= filled_cols || mod(n + parity_offset, 4) == 0
                epsilon[m, n] = n_struct^2
            end
        end
    end
    return epsilon
end

function build_truncated_sources(
    ny::Int,
    ny_L::Int,
    dx::Float64,
    D_in::Float64,
    k0dx::Float64,
    epsilon_L::Float64,
    yBC,
    theta_in_list::Vector{Float64},
    use_continuous_dispersion::Bool,
)
    channels_L = mesti_build_channels(ny, yBC, k0dx, epsilon_L, nothing, use_continuous_dispersion)
    kydx_FOV = k0dx .* sind.(theta_in_list)
    kxdx_FOV = sqrt.((k0dx)^2 .- (kydx_FOV).^2)
    N_L = length(kydx_FOV)

    B_basis = channels_L.f_x_m(channels_L.kydx_prop)
    sqrt_nu_L_basis = reshape(channels_L.sqrt_nu_prop, 1, :)
    y_source = (collect(0.5:1:ny) .- ny / 2) .* dx
    ind_source_out = findall(x -> abs(x) > D_in / 2, y_source)
    B_trunc = channels_L.f_x_m(kydx_FOV) .* sqrt(ny / ny_L)
    B_trunc[ind_source_out, :] .= 0

    B_L = zeros(ComplexF64, ny, N_L)
    for ii = 1:N_L
        for jj = 1:channels_L.N_prop
            Psi_in = sum(B_trunc[:, ii] .* conj(B_basis[:, jj]), dims=1)
            B_L[:, ii] .= B_L[:, ii] .+ Psi_in .* sqrt_nu_L_basis[jj] .* exp((-1im * 1 / 2) * channels_L.kzdx_prop[jj]) .* B_basis[:, jj]
        end
    end

    return channels_L, kydx_FOV, kxdx_FOV, y_source, ind_source_out, B_trunc, B_L
end

function asp_setup(
    dx::Float64,
    wavelength::Float64,
    n_air::Float64,
    D_out::Float64,
    ny_R::Int,
    dy_ASP::Float64,
)
    if round(dy_ASP / dx) != dy_ASP / dx
        dy_ASP = maximum([1, round(dy_ASP / dx)]) * dx
    end
    if mod(ny_R, 2) == 0 && mod(dy_ASP / dx, 2) == 0
        dy_ASP = ((dy_ASP / dx) - 1) * dx
    end

    W_ASP_min = 2 * D_out
    ny_ASP = nextpow(2, Int(round(W_ASP_min / dy_ASP)))
    W_ASP = ny_ASP * dy_ASP
    ind_ASP = Int.(collect(round.(1:(dy_ASP / dx):ny_R)))
    if ind_ASP[end] != ny_R
        if mod(ny_R - ind_ASP[end], 2) != 0
            ind_ASP = ind_ASP[1:(end - 1)]
        end
        ind_ASP = Int.(ind_ASP .+ (ny_R .- ind_ASP[end]) ./ 2)
    end

    ny_ASP_pad = ny_ASP - length(ind_ASP)
    ny_ASP_pad_low = Int(round(ny_ASP_pad / 2))
    ny_ASP_pad_high = ny_ASP_pad - ny_ASP_pad_low
    y_ASP = (collect(0.5:ny_ASP) .- 0.5 * (ny_ASP + ny_ASP_pad_low - ny_ASP_pad_high)) .* dy_ASP

    ny_ASP_half = Int(round(ny_ASP / 2))
    ky_ASP = (2 * pi / W_ASP) .* [collect(0:(ny_ASP_half - 1)); collect(-ny_ASP_half:-1)]
    kx_ASP = sqrt.(Complex.((n_air * 2 * pi / wavelength)^2 .- ky_ASP .^ 2))
    prop_mask = findall(x -> abs(x) < (n_air * 2 * pi / wavelength), ky_ASP)
    ky_ASP_prop = ky_ASP[prop_mask]
    kx_ASP_prop = sqrt.((n_air * 2 * pi / wavelength)^2 .- ky_ASP_prop .^ 2)

    return dy_ASP, ny_ASP, W_ASP, ind_ASP, ny_ASP_pad_low, ny_ASP_pad_high, y_ASP, ky_ASP, kx_ASP, prop_mask, ky_ASP_prop, kx_ASP_prop
end

function main()
    n_air = 1.0
    n_sub = 1.0
    n_struct = 1.45
    wavelength = 1.0
    dx = 0.25
    FOV = 40.0
    theta_in_list = [-20.0, 0.0, 20.0]

    h = 1.0
    D_out = 2.0
    D_in = 1.0
    NA = 0.6
    focal_length = D_out / 2 / tan(asin(NA))
    W_out = D_out + 1.0

    ny_R_extra_half = Int(round((W_out - D_out) / dx / 2))
    ny = Int(ceil(D_out / dx))
    ny_L = Int(ceil(D_in / dx))
    ny_R = ny + 2 * ny_R_extra_half
    nz = Int(ceil(h / dx))

    nPML = 2
    nz_extra_left = 1 + nPML
    nz_extra_right = nz_extra_left
    ny_extra_low = ny_R_extra_half + nPML
    ny_extra_high = ny_extra_low
    ny_tot = ny + ny_extra_low + ny_extra_high
    nz_tot = nz + nz_extra_left + nz_extra_right

    k0dx = 2 * pi / wavelength * dx
    epsilon_L = n_sub^2
    epsilon_R = n_air^2
    yBC_channels = "periodic"
    use_continuous_dispersion = true

    channels_L, kydx_FOV, kxdx_FOV, y_source, ind_source_out, B_trunc, B_L = build_truncated_sources(
        ny,
        ny_L,
        dx,
        D_in,
        k0dx,
        epsilon_L,
        yBC_channels,
        theta_in_list,
        use_continuous_dispersion,
    )

    epsilon_metalens = demo_epsilon_metalens(ny, nz, n_struct)
    epsilon_syst = ones(ComplexF64, ny_tot, nz_tot)
    epsilon_syst[ny_extra_low .+ (1:ny), nz_extra_left .+ (1:nz)] = epsilon_metalens

    syst = Syst()
    syst.epsilon_xx = epsilon_syst
    syst.wavelength = wavelength
    syst.dx = dx
    pml = PML(nPML)
    pml.direction = "all"
    syst.PML = [pml]

    n_L = nz_extra_left
    m1_L = ny_extra_low + 1
    B_struct = Source_struct()
    B_struct.pos = [[m1_L, n_L, ny, 1]]
    B_struct.data = [B_L]

    dy_ASP_input = dx
    dy_ASP, ny_ASP, W_ASP, ind_ASP, ny_ASP_pad_low, ny_ASP_pad_high, y_ASP, ky_ASP, kx_ASP, prop_mask, ky_ASP_prop, kx_ASP_prop = asp_setup(
        dx,
        wavelength,
        n_air,
        D_out,
        ny_R,
        dy_ASP_input,
    )

    n_R = n_L + nz + 1
    m1_R = nPML + 1
    C_R = zeros(ComplexF64, ny_R, length(ind_ASP))
    C_R[CartesianIndex.(ind_ASP, 1:length(ind_ASP))] .= 1
    C_struct = Source_struct()
    C_struct.pos = [[m1_R, n_R, ny_R, 1]]
    C_struct.data = [C_R]

    opts = Opts()
    opts.prefactor = -2im
    opts.verbal = false
    opts.use_L0_threads = false
    opts.use_single_precision_MUMPS = false

    t0 = time()
    field_right_after_metalens, info_direct = mesti(syst, [B_struct], [C_struct], opts)
    elapsed_direct = time() - t0

    selected_angle_indices = collect(1:length(theta_in_list))
    field_at_focal_plane = zeros(ComplexF64, ny_ASP, length(selected_angle_indices))
    t0 = time()
    for (out_idx, source_idx) in enumerate(selected_angle_indices)
        Ex0_ASP = field_right_after_metalens[:, source_idx]
        field_at_focal_plane[:, out_idx] = asp(Ex0_ASP, focal_length, kx_ASP_prop, ny_ASP)
    end
    elapsed_asp = time() - t0

    focal_spot_list = focal_length .* tand.(theta_in_list)
    target_focal_indices = [argmin(abs.(y_ASP .- spot)) for spot in focal_spot_list]
    target_focal_intensities = [
        abs2(field_at_focal_plane[target_focal_indices[ii], ii])
        for ii in 1:length(selected_angle_indices)
    ]
    focal_plane_intensity = abs2.(field_at_focal_plane)
    peak_intensities = vec(maximum(focal_plane_intensity, dims=1))
    peak_indices = [argmax(focal_plane_intensity[:, ii]) for ii in 1:size(focal_plane_intensity, 2)]
    peak_y_positions = y_ASP[peak_indices]

    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_example_metalens_asp_v5_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "Reduced packaged example fixture for 2D metalens focusing via angular spectrum propagation.",
        "use_single_precision_MUMPS" => false,
        "n_air" => n_air,
        "n_sub" => n_sub,
        "n_struct" => n_struct,
        "wavelength" => wavelength,
        "dx" => dx,
        "FOV" => FOV,
        "theta_in_list" => theta_in_list,
        "h" => h,
        "D_out" => D_out,
        "D_in" => D_in,
        "NA" => NA,
        "focal_length" => focal_length,
        "W_out" => W_out,
        "ny_R_extra_half" => ny_R_extra_half,
        "ny" => ny,
        "ny_L" => ny_L,
        "ny_R" => ny_R,
        "nz" => nz,
        "nPML" => nPML,
        "nz_extra_left" => nz_extra_left,
        "nz_extra_right" => nz_extra_right,
        "ny_extra_low" => ny_extra_low,
        "ny_extra_high" => ny_extra_high,
        "ny_tot" => ny_tot,
        "nz_tot" => nz_tot,
        "k0dx" => k0dx,
        "epsilon_L" => epsilon_L,
        "epsilon_R" => epsilon_R,
        "yBC_channels" => yBC_channels,
        "use_continuous_dispersion" => use_continuous_dispersion,
        "epsilon_metalens" => epsilon_metalens,
        "epsilon_syst" => epsilon_syst,
        "kydx_FOV" => kydx_FOV,
        "kxdx_FOV" => kxdx_FOV,
        "y_source" => y_source,
        "ind_source_out_julia" => ind_source_out,
        "ind_source_out_zero_based" => ind_source_out .- 1,
        "B_basis" => channels_L.f_x_m(channels_L.kydx_prop),
        "B_trunc" => B_trunc,
        "B_L" => B_L,
        "N_prop_L" => channels_L.N_prop,
        "kydx_prop_L" => channels_L.kydx_prop,
        "kzdx_prop_L" => channels_L.kzdx_prop,
        "sqrt_nu_prop_L" => channels_L.sqrt_nu_prop,
        "source_pos_julia" => [m1_L, n_L, ny, 1],
        "source_pos_zero_based_inclusive" => [m1_L - 1, n_L - 1, m1_L + ny - 2, n_L - 1],
        "projection_pos_julia" => [m1_R, n_R, ny_R, 1],
        "projection_pos_zero_based_inclusive" => [m1_R - 1, n_R - 1, m1_R + ny_R - 2, n_R - 1],
        "dy_ASP_input" => dy_ASP_input,
        "dy_ASP" => dy_ASP,
        "ny_ASP" => ny_ASP,
        "W_ASP" => W_ASP,
        "ind_ASP_julia" => ind_ASP,
        "ind_ASP_zero_based" => ind_ASP .- 1,
        "ny_ASP_pad_low" => ny_ASP_pad_low,
        "ny_ASP_pad_high" => ny_ASP_pad_high,
        "y_ASP" => y_ASP,
        "ky_ASP" => ky_ASP,
        "kx_ASP" => kx_ASP,
        "asp_prop_indices_julia" => prop_mask,
        "asp_prop_indices_zero_based" => prop_mask .- 1,
        "ky_ASP_prop" => ky_ASP_prop,
        "kx_ASP_prop" => kx_ASP_prop,
        "C_R" => C_R,
        "field_right_after_metalens" => field_right_after_metalens,
        "selected_angle_indices_julia" => selected_angle_indices,
        "selected_angle_indices_zero_based" => selected_angle_indices .- 1,
        "field_at_focal_plane" => field_at_focal_plane,
        "focal_plane_intensity" => focal_plane_intensity,
        "focal_spot_list" => focal_spot_list,
        "target_focal_indices_julia" => target_focal_indices,
        "target_focal_indices_zero_based" => target_focal_indices .- 1,
        "target_focal_intensities" => target_focal_intensities,
        "peak_intensities" => peak_intensities,
        "peak_indices_julia" => peak_indices,
        "peak_indices_zero_based" => peak_indices .- 1,
        "peak_y_positions" => peak_y_positions,
        "return_field_profile_direct" => info_direct.opts.return_field_profile,
        "elapsed_direct_seconds" => elapsed_direct,
        "elapsed_asp_seconds" => elapsed_asp,
    )

    matwrite(OUT_PATH, payload)
    println("Recorded reduced metalens ASP example with epsilon_syst size ", size(epsilon_syst))
    println("B_L size = ", size(B_L), ", N_prop_L = ", channels_L.N_prop)
    println("field right-after-metalens size = ", size(field_right_after_metalens))
    println("field at focal plane size = ", size(field_at_focal_plane))
    println("target focal intensities = ", target_focal_intensities)
    println("peak intensities = ", peak_intensities)
    println("Wrote ", OUT_PATH)
end

main()
