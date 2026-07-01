"""Generate v4 Julia parity fixtures for 3D diagonal ``mesti2s``.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_3d_diagonal_julia_fixtures.jl'

The fixtures are intentionally small and use periodic transverse boundaries
with no PML pixels. They lock down the 3D diagonal tensor channel ordering,
s/p-polarized surface source/projection assembly, channel subselects, mixed
wavefront field profiles, and one-sided low reflection.
"""

using LinearAlgebra
using MAT
using MESTI

const OUT_PATH = joinpath(@__DIR__, "mesti2s_3d_diagonal_periodic.mat")

function patterned_epsilon(nx, ny, nz, base)
    epsilon = zeros(ComplexF64, nx, ny, nz)
    for iz in 1:nz
        for iy in 1:ny
            for ix in 1:nx
                epsilon[ix, iy, iz] =
                    base +
                    0.035 * ix +
                    0.019 * iy +
                    0.011 * iz +
                    im * (0.004 * ix - 0.003 * iy + 0.002 * iz)
            end
        end
    end
    return epsilon
end

function make_two_sided_system()
    syst = Syst()
    syst.epsilon_xx = patterned_epsilon(3, 3, 1, 1.00)
    syst.epsilon_yy = patterned_epsilon(3, 3, 1, 1.08)
    syst.epsilon_zz = patterned_epsilon(3, 3, 2, 1.16)
    syst.epsilon_low = 1.0
    syst.epsilon_high = 1.0
    syst.wavelength = 2 * pi / 1.8
    syst.dx = 1.0
    syst.xBC = "periodic"
    syst.yBC = "periodic"
    syst.zPML = [PML(0)]
    return syst
end

function make_one_sided_system()
    syst = Syst()
    syst.epsilon_xx = patterned_epsilon(3, 3, 1, 0.96)
    syst.epsilon_yy = patterned_epsilon(3, 3, 1, 1.04)
    syst.epsilon_zz = patterned_epsilon(3, 3, 2, 1.12)
    syst.epsilon_low = 1.0
    syst.wavelength = 2 * pi / 1.8
    syst.dx = 1.0
    syst.xBC = "periodic"
    syst.yBC = "periodic"
    syst.zPML = [PML(0)]
    return syst
end

function base_opts()
    opts = Opts()
    opts.verbal = false
    opts.solver = "JULIA"
    opts.use_single_precision_MUMPS = false
    return opts
end

function all_channels(side)
    spec = channel_type()
    spec.side = side
    spec.polarization = "both"
    return spec
end

function wave_matrix(n, m; seed=1.0)
    out = zeros(ComplexF64, n, m)
    for col in 1:m
        for row in 1:n
            out[row, col] =
                seed * (0.09 * row - 0.04 * col) +
                im * seed * (0.05 * row + 0.025 * col)
        end
    end
    return out
end

function channel_payload(prefix, side)
    return Dict{String,Any}(
        "$(prefix)_N_prop" => side.N_prop,
        "$(prefix)_ind_prop_julia" => side.ind_prop,
        "$(prefix)_ind_prop_zero_based" => side.ind_prop .- 1,
        "$(prefix)_kxdx_prop" => side.kxdx_prop,
        "$(prefix)_kydx_prop" => side.kydx_prop,
        "$(prefix)_kzdx_prop" => side.kzdx_prop,
        "$(prefix)_sqrt_nu_prop" => side.sqrt_nu_prop,
    )
end

function two_sided_channel_payload(channels)
    payload = Dict{String,Any}(
        "kxdx_all" => channels.kxdx_all,
        "kydx_all" => channels.kydx_all,
    )
    merge!(payload, channel_payload("low", channels.low))
    merge!(payload, channel_payload("high", channels.high))
    return payload
end

function one_sided_channel_payload(channels)
    payload = Dict{String,Any}(
        "one_kxdx_all" => channels.kxdx_all,
        "one_kydx_all" => channels.kydx_all,
    )
    merge!(payload, channel_payload("one_low", channels))
    return payload
end

function paired_modes(n_func, m_func, kxdx, kydx)
    f_n = n_func(kxdx)
    f_m = m_func(kydx)
    out = zeros(ComplexF64, size(f_n, 1) * size(f_m, 1), size(f_n, 2))
    for col in 1:size(f_n, 2)
        out[:, col] = reshape(f_n[:, col] * transpose(f_m[:, col]), :)
    end
    return out
end

function one_sided_low_reflection_manual(syst)
    # The Julia 0.5.1 high-level 3D one-sided mesti2s path references
    # N_prop_high before it is defined. This fixture keeps one-sided coverage by
    # using the same Julia channel and direct-mesti formulas explicitly.
    k0dx = (2 * pi / syst.wavelength) * syst.dx
    nx_Ex, ny_Ex, nz_Ex = size(syst.epsilon_xx)
    nx_Ey, ny_Ey, nz_Ey = size(syst.epsilon_yy)
    nx_Ez, ny_Ez, nz_Ez = size(syst.epsilon_zz)
    channels = mesti_build_channels(
        nx_Ex,
        nx_Ey,
        syst.xBC,
        ny_Ex,
        ny_Ey,
        syst.yBC,
        k0dx,
        syst.epsilon_low,
        nothing,
        false,
        0,
        0,
    )
    N = channels.N_prop
    ind = 1:N
    dn = 0.5

    f_Ex = paired_modes(channels.f_x_n, channels.f_x_m, channels.kxdx_prop, channels.kydx_prop)
    f_Ey = paired_modes(channels.f_y_n, channels.f_y_m, channels.kxdx_prop, channels.kydx_prop)
    f_dEz_dx = paired_modes(channels.df_z_n, channels.f_z_m, channels.kxdx_prop, channels.kydx_prop)
    f_dEz_dy = paired_modes(channels.f_z_n, channels.df_z_m, channels.kxdx_prop, channels.kydx_prop)

    kappa_x = sin.(channels.kxdx_prop / 2)
    kappa_y = sin.(channels.kydx_prop / 2)
    kappa_z = sin.(channels.kzdx_prop / 2)

    denominator_s = sqrt.(kappa_x .^ 2 + kappa_y .^ 2)
    alpha_x_s = -kappa_y ./ denominator_s
    alpha_y_s = kappa_x ./ denominator_s
    alpha_x_s[isnan.(alpha_x_s)] .= 0
    alpha_y_s[isnan.(alpha_y_s)] .= 1

    denominator_p = sqrt.(
        (abs.(kappa_x .* kappa_z)) .^ 2 .+
        (abs.(kappa_y .* kappa_z)) .^ 2 .+
        (abs.(kappa_x .^ 2 + kappa_y .^ 2)) .^ 2
    )
    alpha_x_p = kappa_x .* kappa_z ./ denominator_p
    alpha_y_p = kappa_y .* kappa_z ./ denominator_p
    alpha_z_p = -(kappa_x .^ 2 + kappa_y .^ 2) ./ denominator_p
    alpha_x_p[isnan.(alpha_x_p)] .= 1
    alpha_y_p[isnan.(alpha_y_p)] .= 0
    alpha_z_p[isnan.(alpha_z_p)] .= 0

    sqrt_nu = reshape(channels.sqrt_nu_prop, 1, :)
    dz_weight = reshape(cos.(channels.kzdx_prop / 2) ./ channels.sqrt_nu_prop, 1, :)

    B_s_Ex = f_Ex[:, ind] .* sqrt_nu .* reshape(alpha_x_s[ind], 1, :)
    B_s_Ey = f_Ey[:, ind] .* sqrt_nu .* reshape(alpha_y_s[ind], 1, :)
    B_p_Ex = f_Ex[:, ind] .* sqrt_nu .* reshape(alpha_x_p[ind], 1, :)
    B_p_Ey = f_Ey[:, ind] .* sqrt_nu .* reshape(alpha_y_p[ind], 1, :)
    B_p_dx = f_dEz_dx[:, ind] .* dz_weight .* reshape(alpha_z_p[ind], 1, :)
    B_p_dy = f_dEz_dy[:, ind] .* dz_weight .* reshape(alpha_z_p[ind], 1, :)
    B_Ex_low = reshape([B_s_Ex B_p_Ex + 1im * B_p_dx], nx_Ex, ny_Ex, 1, :)
    B_Ey_low = reshape([B_s_Ey B_p_Ey + 1im * B_p_dy], nx_Ey, ny_Ey, 1, :)

    C_s_Ex = conj.(f_Ex[:, ind]) .* sqrt_nu .* conj.(reshape(alpha_x_s[ind], 1, :))
    C_s_Ey = conj.(f_Ey[:, ind]) .* sqrt_nu .* conj.(reshape(alpha_y_s[ind], 1, :))
    C_p_Ex = conj.(f_Ex[:, ind]) .* sqrt_nu .* conj.(reshape(alpha_x_p[ind], 1, :))
    C_p_Ey = conj.(f_Ey[:, ind]) .* sqrt_nu .* conj.(reshape(alpha_y_p[ind], 1, :))
    C_p_dx = conj.(f_dEz_dx[:, ind]) .* dz_weight .* conj.(reshape(alpha_z_p[ind], 1, :))
    C_p_dy = conj.(f_dEz_dy[:, ind]) .* dz_weight .* conj.(reshape(alpha_z_p[ind], 1, :))
    C_Ex_low = reshape([C_s_Ex C_p_Ex - 1im * C_p_dx], nx_Ex, ny_Ex, 1, :)
    C_Ey_low = reshape([C_s_Ey C_p_Ey - 1im * C_p_dy], nx_Ey, ny_Ey, 1, :)

    nz_extra_low = 1 + syst.zPML[1].npixels
    padded = deepcopy(syst)
    padded.epsilon_xx = cat(syst.epsilon_low * ones(nx_Ex, ny_Ex, nz_extra_low), syst.epsilon_xx, dims=3)
    padded.epsilon_yy = cat(syst.epsilon_low * ones(nx_Ey, ny_Ey, nz_extra_low), syst.epsilon_yy, dims=3)
    padded.epsilon_zz = cat(syst.epsilon_low * ones(nx_Ez, ny_Ez, nz_extra_low), syst.epsilon_zz, dims=3)
    padded.epsilon_low = nothing
    padded.epsilon_high = nothing
    low_pml = deepcopy(syst.zPML[1])
    low_pml.direction = "z"
    low_pml.side = "-"
    low_pml.npixels_spacer = nothing
    padded.PML = [low_pml]
    padded.zPML = nothing
    padded.zBC = "PEC"

    l_low = nz_extra_low
    B_Ex = Source_struct()
    B_Ey = Source_struct()
    B_Ez = Source_struct()
    B_Ez.isempty = true
    B_Ex.pos = [[1, 1, l_low, nx_Ex, ny_Ex, 1]]
    B_Ey.pos = [[1, 1, l_low, nx_Ey, ny_Ey, 1]]
    B_Ex.data = [B_Ex_low]
    B_Ey.data = [B_Ey_low]

    C_Ex = Source_struct()
    C_Ey = Source_struct()
    C_Ez = Source_struct()
    C_Ez.isempty = true
    C_Ex.pos = [[1, 1, l_low, nx_Ex, ny_Ex, 1]]
    C_Ey.pos = [[1, 1, l_low, nx_Ey, ny_Ey, 1]]
    C_Ex.data = [C_Ex_low]
    C_Ey.data = [C_Ey_low]

    opts = base_opts()
    S, info = mesti(padded, [B_Ex, B_Ey, B_Ez], [C_Ex, C_Ey, C_Ez], nothing, opts)
    S = (-2im) * S
    prefactor = [exp.((-1im * dn) * channels.kzdx_prop); exp.((-1im * dn) * channels.kzdx_prop)]
    S = prefactor .* S .* reshape(prefactor, 1, :)
    D = Diagonal([exp.((-1im * 2 * dn) * channels.kzdx_prop); exp.((-1im * 2 * dn) * channels.kzdx_prop)])
    S = S - D
    return S, channels, info
end

function main()
    two = make_two_sided_system()
    opts = base_opts()

    both = all_channels("both")
    S_both, channels, info_s = mesti2s(two, both, both, opts)

    subset_in = channel_index()
    subset_in.ind_low_s = [1, 3]
    subset_in.ind_low_p = [2]
    subset_in.ind_high_s = [4]
    subset_in.ind_high_p = [1]
    subset_out = channel_index()
    subset_out.ind_low_p = [1]
    subset_out.ind_high_s = [2, 5]
    subset_out.ind_high_p = [3]
    S_subset, _, info_subset = mesti2s(two, subset_in, subset_out, base_opts())

    wf = wavefront()
    wf.v_low_s = wave_matrix(channels.low.N_prop, 2; seed=1.0)
    wf.v_low_p = wave_matrix(channels.low.N_prop, 1; seed=-0.65)
    wf.v_high_p = wave_matrix(channels.high.N_prop, 1; seed=0.45)
    Ex_field, Ey_field, Ez_field, _, info_field = mesti2s(two, wf, base_opts())

    one = make_one_sided_system()
    S_one, channels_one, info_one = one_sided_low_reflection_manual(one)

    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_mesti2s_3d_diagonal_julia_fixtures.jl",
        "julia_mesti_version" => "0.5.1",
        "solver" => "JULIA",
        "use_single_precision_MUMPS" => false,
        "description" => "V4 3D diagonal mesti2s periodic fixture for channels, s/p scattering, subselects, wavefront fields, and one-sided reflection.",
        "epsilon_xx" => two.epsilon_xx,
        "epsilon_yy" => two.epsilon_yy,
        "epsilon_zz" => two.epsilon_zz,
        "epsilon_low" => two.epsilon_low,
        "epsilon_high" => two.epsilon_high,
        "wavelength" => two.wavelength,
        "dx" => two.dx,
        "xBC" => two.xBC,
        "yBC" => two.yBC,
        "zPML_npixels" => two.zPML[1].npixels,
        "S_both" => S_both,
        "S_both_singular_values" => svd(S_both).S,
        "return_field_profile_S" => info_s.opts.return_field_profile,
        "subset_in_low_s_julia" => subset_in.ind_low_s,
        "subset_in_low_p_julia" => subset_in.ind_low_p,
        "subset_in_high_s_julia" => subset_in.ind_high_s,
        "subset_in_high_p_julia" => subset_in.ind_high_p,
        "subset_out_low_p_julia" => subset_out.ind_low_p,
        "subset_out_high_s_julia" => subset_out.ind_high_s,
        "subset_out_high_p_julia" => subset_out.ind_high_p,
        "subset_in_low_s_zero_based" => subset_in.ind_low_s .- 1,
        "subset_in_low_p_zero_based" => subset_in.ind_low_p .- 1,
        "subset_in_high_s_zero_based" => subset_in.ind_high_s .- 1,
        "subset_in_high_p_zero_based" => subset_in.ind_high_p .- 1,
        "subset_out_low_p_zero_based" => subset_out.ind_low_p .- 1,
        "subset_out_high_s_zero_based" => subset_out.ind_high_s .- 1,
        "subset_out_high_p_zero_based" => subset_out.ind_high_p .- 1,
        "S_subset" => S_subset,
        "return_field_profile_subset" => info_subset.opts.return_field_profile,
        "v_low_s" => wf.v_low_s,
        "v_low_p" => wf.v_low_p,
        "v_high_p" => wf.v_high_p,
        "field_Ex" => Ex_field,
        "field_Ey" => Ey_field,
        "field_Ez" => Ez_field,
        "return_field_profile_field" => info_field.opts.return_field_profile,
        "one_epsilon_xx" => one.epsilon_xx,
        "one_epsilon_yy" => one.epsilon_yy,
        "one_epsilon_zz" => one.epsilon_zz,
        "one_epsilon_low" => one.epsilon_low,
        "one_wavelength" => one.wavelength,
        "one_dx" => one.dx,
        "one_xBC" => one.xBC,
        "one_yBC" => one.yBC,
        "one_zPML_npixels" => one.zPML[1].npixels,
        "S_one_low" => S_one,
        "S_one_low_singular_values" => svd(S_one).S,
        "return_field_profile_one" => info_one.opts.return_field_profile,
    )
    merge!(payload, two_sided_channel_payload(channels))
    merge!(payload, one_sided_channel_payload(channels_one))
    matwrite(OUT_PATH, payload)
    println("Wrote ", OUT_PATH, " with two-sided N_prop=", channels.low.N_prop, " and one-sided N_prop=", channels_one.N_prop)
end

main()
