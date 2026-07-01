"""Generate v5 Julia-reference fixtures for 3D ``mesti2s`` nz extension.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_3d_nz_extension_julia_fixture.jl'

Julia MESTI 0.5.1 has broken high-level code in the 3D
``opts.nz_low``/``opts.nz_high`` extension branch.  This generator therefore
uses high-level Julia ``mesti2s`` only to obtain the verified source-surface
field profiles with ``nz_low = nz_high = 1`` for two-sided systems, then applies
the corrected homogeneous-region formulas from ``src/mesti2s.jl`` locally.  The
one-sided fixture uses the same low-level source construction as the v4 manual
one-sided reference because upstream one-sided 3D ``mesti2s`` references
``N_prop_high`` before it exists.
"""

using LinearAlgebra
using MAT
using MESTI

const OUT_PATH = joinpath(@__DIR__, "mesti2s_3d_nz_extension.mat")
const DN = 0.5

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

function surface_opts(two_sided)
    opts = base_opts()
    opts.nz_low = 1
    opts.nz_high = two_sided ? 1 : 0
    return opts
end

function all_channels()
    spec = channel_type()
    spec.side = "both"
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

function build_channels_for_system(syst; two_sided=true)
    k0dx = (2 * pi / syst.wavelength) * syst.dx
    nx_Ex, ny_Ex, _ = size(syst.epsilon_xx)
    nx_Ey, ny_Ey, _ = size(syst.epsilon_yy)
    return mesti_build_channels(
        nx_Ex,
        nx_Ey,
        syst.xBC,
        ny_Ex,
        ny_Ey,
        syst.yBC,
        k0dx,
        syst.epsilon_low,
        two_sided ? syst.epsilon_high : nothing,
        false,
        0,
        0,
    )
end

function channel_payload!(payload, prefix, side)
    payload["$(prefix)_N_prop"] = side.N_prop
    payload["$(prefix)_ind_prop_julia"] = side.ind_prop
    payload["$(prefix)_ind_prop_zero_based"] = side.ind_prop .- 1
    payload["$(prefix)_kxdx_prop"] = side.kxdx_prop
    payload["$(prefix)_kydx_prop"] = side.kydx_prop
    payload["$(prefix)_kzdx_prop"] = side.kzdx_prop
    payload["$(prefix)_sqrt_nu_prop"] = side.sqrt_nu_prop
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

function all_modes(channels)
    return (
        Ex = kron(channels.f_x_m(channels.kydx_all), channels.f_x_n(channels.kxdx_all)),
        Ey = kron(channels.f_y_m(channels.kydx_all), channels.f_y_n(channels.kxdx_all)),
        Ez = kron(channels.f_z_m(channels.kydx_all), channels.f_z_n(channels.kxdx_all)),
    )
end

function prop_modes(channels, side)
    return (
        Ex = paired_modes(channels.f_x_n, channels.f_x_m, side.kxdx_prop, side.kydx_prop),
        Ey = paired_modes(channels.f_y_n, channels.f_y_m, side.kxdx_prop, side.kydx_prop),
        dEz_dx = paired_modes(channels.df_z_n, channels.f_z_m, side.kxdx_prop, side.kydx_prop),
        dEz_dy = paired_modes(channels.f_z_n, channels.df_z_m, side.kxdx_prop, side.kydx_prop),
    )
end

function prop_alpha(side)
    kappa_x = sin.(side.kxdx_prop / 2)
    kappa_y = sin.(side.kydx_prop / 2)
    kappa_z = sin.(side.kzdx_prop / 2)

    denominator_s = sqrt.(kappa_x .^ 2 .+ kappa_y .^ 2)
    alpha_x_s = -kappa_y ./ denominator_s
    alpha_y_s = kappa_x ./ denominator_s
    alpha_x_s[isnan.(alpha_x_s)] .= 0
    alpha_y_s[isnan.(alpha_y_s)] .= 1

    denominator_p = sqrt.(
        (abs.(kappa_x .* kappa_z)) .^ 2 .+
        (abs.(kappa_y .* kappa_z)) .^ 2 .+
        (abs.(kappa_x .^ 2 .+ kappa_y .^ 2)) .^ 2
    )
    alpha_x_p = kappa_x .* kappa_z ./ denominator_p
    alpha_y_p = kappa_y .* kappa_z ./ denominator_p
    alpha_z_p = -(kappa_x .^ 2 .+ kappa_y .^ 2) ./ denominator_p
    alpha_x_p[isnan.(alpha_x_p)] .= 1
    alpha_y_p[isnan.(alpha_y_p)] .= 0
    alpha_z_p[isnan.(alpha_z_p)] .= 0
    return (
        x_s = alpha_x_s,
        y_s = alpha_y_s,
        x_p = alpha_x_p,
        y_p = alpha_y_p,
        z_p = alpha_z_p,
    )
end

function all_alpha(channels, side, direction)
    kxdx_all = vec(channels.kxdx_all)
    kydx_all = vec(channels.kydx_all)
    kappa_x = sin.(reshape(repeat(kxdx_all, 1, length(kydx_all)), :) / 2)
    kappa_y = sin.(reshape(repeat(transpose(kydx_all), length(kxdx_all), 1), :) / 2)
    kappa_z = sin.(vec(side.kzdx_all) / 2)

    denominator_s = sqrt.(kappa_x .^ 2 .+ kappa_y .^ 2)
    alpha_x_s = -kappa_y ./ denominator_s
    alpha_y_s = kappa_x ./ denominator_s
    alpha_x_s[isnan.(alpha_x_s)] .= 0
    alpha_y_s[isnan.(alpha_y_s)] .= 1

    denominator_p = sqrt.(
        (abs.(kappa_x .* kappa_z)) .^ 2 .+
        (abs.(kappa_y .* kappa_z)) .^ 2 .+
        (abs.(kappa_x .^ 2 .+ kappa_y .^ 2)) .^ 2
    )
    alpha_x_p = kappa_x .* kappa_z ./ denominator_p
    alpha_y_p = kappa_y .* kappa_z ./ denominator_p
    z_sign = direction == "low" ? 1 : -1
    alpha_z_p = z_sign .* (kappa_x .^ 2 .+ kappa_y .^ 2) ./ denominator_p
    alpha_x_p[isnan.(alpha_x_p)] .= 1
    alpha_y_p[isnan.(alpha_y_p)] .= 0
    alpha_z_p[isnan.(alpha_z_p)] .= 0
    return (
        x_s = alpha_x_s,
        y_s = alpha_y_s,
        z_s = zeros(ComplexF64, length(kappa_z)),
        x_p = alpha_x_p,
        y_p = alpha_y_p,
        z_p = alpha_z_p,
    )
end

function basis_matrix(modes, alpha, polarization)
    if polarization == "s"
        return [
            modes.Ex .* reshape(alpha.x_s, 1, :)
            modes.Ey .* reshape(alpha.y_s, 1, :)
            modes.Ez .* reshape(alpha.z_s, 1, :)
        ]
    end
    return [
        modes.Ex .* reshape(alpha.x_p, 1, :)
        modes.Ey .* reshape(alpha.y_p, 1, :)
        modes.Ez .* reshape(alpha.z_p, 1, :)
    ]
end

function project_surface(Ex_slice, Ey_slice, Ez_slice, basis_s, basis_p)
    stacked = [vec(Ex_slice); vec(Ey_slice); vec(Ez_slice)]
    return basis_s' * stacked, basis_p' * stacked
end

function synthesize_layers(modes, alpha, coeff_s, coeff_p, shape_ex, shape_ey, shape_ez)
    nz = size(coeff_s, 2)
    Ex_vec =
        (modes.Ex .* reshape(alpha.x_s, 1, :)) * coeff_s +
        (modes.Ex .* reshape(alpha.x_p, 1, :)) * coeff_p
    Ey_vec =
        (modes.Ey .* reshape(alpha.y_s, 1, :)) * coeff_s +
        (modes.Ey .* reshape(alpha.y_p, 1, :)) * coeff_p
    Ez_vec =
        (modes.Ez .* reshape(alpha.z_s, 1, :)) * coeff_s +
        (modes.Ez .* reshape(alpha.z_p, 1, :)) * coeff_p
    return (
        reshape(Ex_vec, shape_ex[1], shape_ex[2], nz),
        reshape(Ey_vec, shape_ey[1], shape_ey[2], nz),
        reshape(Ez_vec, shape_ez[1], shape_ez[2], nz),
    )
end

function input_columns(input_info)
    columns = Tuple{Int,String,String,Int}[]
    col = 1
    for side in ("low", "high")
        for pol in ("s", "p")
            key = "$(side)_$(pol)"
            count = input_info["use_indices"] ? length(input_info["ind_$(key)"]) : size(input_info["v_$(key)"], 2)
            for local_index in 1:count
                push!(columns, (col, side, pol, local_index))
                col += 1
            end
        end
    end
    return columns
end

function incident_coefficients(input_info, side, side_key, pol, local_index)
    n_all = length(side.kzdx_all)
    c_s = zeros(ComplexF64, n_all)
    c_p = zeros(ComplexF64, n_all)
    if input_info["use_indices"]
        prop_index = input_info["ind_$(side_key)_$(pol)"][local_index]
        all_index = side.ind_prop[prop_index]
        value = exp((-1im * DN) * side.kzdx_prop[prop_index]) / side.sqrt_nu_prop[prop_index]
        if pol == "s"
            c_s[all_index] = value
        else
            c_p[all_index] = value
        end
    else
        values =
            exp.((-1im * DN) .* side.kzdx_prop) ./
            side.sqrt_nu_prop .*
            input_info["v_$(side_key)_$(pol)"][:, local_index]
        if pol == "s"
            c_s[side.ind_prop] = values
        else
            c_p[side.ind_prop] = values
        end
    end
    return c_s, c_p
end

function extend_low(Ex, Ey, Ez, channels, input_info, modes, alpha, nz_low_extra)
    low = isa(channels, Channels_two_sided) ? channels.low : channels
    basis_s = basis_matrix(modes, alpha, "s")
    basis_p = basis_matrix(modes, alpha, "p")
    l = collect((-nz_low_extra):1:-1)
    phase = reshape(low.kzdx_all, :, 1) .* reshape(l, 1, :)
    exp_pikz = exp.(1im .* phase)
    exp_mikz = exp.(-1im .* phase)
    shape_ex = size(Ex)[1:2]
    shape_ey = size(Ey)[1:2]
    shape_ez = size(Ez)[1:2]
    Ex_low = zeros(ComplexF64, shape_ex[1], shape_ex[2], nz_low_extra, size(Ex, 4))
    Ey_low = zeros(ComplexF64, shape_ey[1], shape_ey[2], nz_low_extra, size(Ey, 4))
    Ez_low = zeros(ComplexF64, shape_ez[1], shape_ez[2], nz_low_extra, size(Ez, 4))

    for (col, side_key, pol, local_index) in input_columns(input_info)
        c_s, c_p = project_surface(Ex[:, :, 1, col], Ey[:, :, 1, col], Ez[:, :, 1, col], basis_s, basis_p)
        if side_key == "low"
            c_in_s, c_in_p = incident_coefficients(input_info, low, "low", pol, local_index)
        else
            c_in_s = zeros(ComplexF64, length(c_s))
            c_in_p = zeros(ComplexF64, length(c_p))
        end
        coeff_s = reshape(c_in_s, :, 1) .* exp_pikz .+ reshape(c_s - c_in_s, :, 1) .* exp_mikz
        coeff_p = reshape(c_in_p, :, 1) .* exp_pikz .+ reshape(c_p - c_in_p, :, 1) .* exp_mikz
        Ex_low[:, :, :, col], Ey_low[:, :, :, col], Ez_low[:, :, :, col] =
            synthesize_layers(modes, alpha, coeff_s, coeff_p, shape_ex, shape_ey, shape_ez)
    end
    return cat(Ex_low, Ex, dims=3), cat(Ey_low, Ey, dims=3), cat(Ez_low, Ez, dims=3)
end

function extend_high(Ex, Ey, Ez, channels, input_info, modes, alpha, nz_high_extra)
    high = channels.high
    basis_s = basis_matrix(modes, alpha, "s")
    basis_p = basis_matrix(modes, alpha, "p")
    l = collect(1:nz_high_extra)
    phase = reshape(high.kzdx_all, :, 1) .* reshape(l, 1, :)
    exp_pikz = exp.(1im .* phase)
    exp_mikz = exp.(-1im .* phase)
    ex_high = size(Ex, 3)
    ey_high = size(Ey, 3)
    ez_high = ex_high + 1
    shape_ex = size(Ex)[1:2]
    shape_ey = size(Ey)[1:2]
    shape_ez = size(Ez)[1:2]
    Ex_high = zeros(ComplexF64, shape_ex[1], shape_ex[2], nz_high_extra, size(Ex, 4))
    Ey_high = zeros(ComplexF64, shape_ey[1], shape_ey[2], nz_high_extra, size(Ey, 4))
    Ez_high = zeros(ComplexF64, shape_ez[1], shape_ez[2], nz_high_extra, size(Ez, 4))

    for (col, side_key, pol, local_index) in input_columns(input_info)
        c_s, c_p = project_surface(
            Ex[:, :, ex_high, col],
            Ey[:, :, ey_high, col],
            Ez[:, :, ez_high, col],
            basis_s,
            basis_p,
        )
        if side_key == "high"
            c_in_s, c_in_p = incident_coefficients(input_info, high, "high", pol, local_index)
        else
            c_in_s = zeros(ComplexF64, length(c_s))
            c_in_p = zeros(ComplexF64, length(c_p))
        end
        coeff_s = reshape(c_in_s, :, 1) .* exp_mikz .+ reshape(c_s - c_in_s, :, 1) .* exp_pikz
        coeff_p = reshape(c_in_p, :, 1) .* exp_mikz .+ reshape(c_p - c_in_p, :, 1) .* exp_pikz
        Ex_high[:, :, :, col], Ey_high[:, :, :, col], Ez_high[:, :, :, col] =
            synthesize_layers(modes, alpha, coeff_s, coeff_p, shape_ex, shape_ey, shape_ez)
    end
    return cat(Ex, Ex_high, dims=3), cat(Ey, Ey_high, dims=3), cat(Ez, Ez_high, dims=3)
end

function extend_profile_3d(Ex, Ey, Ez, channels, input_info, nz_low, nz_high; two_sided=true)
    modes = nothing
    nz_low_extra = nz_low - 1
    if nz_low_extra == -1
        Ex = Ex[:, :, 2:end, :]
        Ey = Ey[:, :, 2:end, :]
        Ez = Ez[:, :, 2:end, :]
    elseif nz_low_extra > 0
        modes = all_modes(channels)
        alpha_low = all_alpha(channels, isa(channels, Channels_two_sided) ? channels.low : channels, "low")
        Ex, Ey, Ez = extend_low(Ex, Ey, Ez, channels, input_info, modes, alpha_low, nz_low_extra)
    end

    if two_sided
        nz_high_extra = nz_high - 1
        if nz_high_extra == -1
            Ex = Ex[:, :, 1:(end-1), :]
            Ey = Ey[:, :, 1:(end-1), :]
            Ez = Ez[:, :, 1:(end-1), :]
        elseif nz_high_extra > 0
            if modes === nothing
                modes = all_modes(channels)
            end
            alpha_high = all_alpha(channels, channels.high, "high")
            Ex, Ey, Ez = extend_high(Ex, Ey, Ez, channels, input_info, modes, alpha_high, nz_high_extra)
        end
    elseif nz_high > 0
        Ex = cat(Ex, zeros(ComplexF64, size(Ex, 1), size(Ex, 2), nz_high, size(Ex, 4)), dims=3)
        Ey = cat(Ey, zeros(ComplexF64, size(Ey, 1), size(Ey, 2), nz_high, size(Ey, 4)), dims=3)
        Ez = cat(Ez, zeros(ComplexF64, size(Ez, 1), size(Ez, 2), nz_high, size(Ez, 4)), dims=3)
    end
    return Ex, Ey, Ez
end

function index_input_info(channels; two_sided=true)
    if two_sided
        return Dict{String,Any}(
            "use_indices" => true,
            "ind_low_s" => collect(1:channels.low.N_prop),
            "ind_low_p" => collect(1:channels.low.N_prop),
            "ind_high_s" => collect(1:channels.high.N_prop),
            "ind_high_p" => collect(1:channels.high.N_prop),
        )
    end
    return Dict{String,Any}(
        "use_indices" => true,
        "ind_low_s" => collect(1:channels.N_prop),
        "ind_low_p" => collect(1:channels.N_prop),
        "ind_high_s" => Int[],
        "ind_high_p" => Int[],
    )
end

function wavefront_input_info(wf, channels; two_sided=true)
    n_low = two_sided ? channels.low.N_prop : channels.N_prop
    n_high = two_sided ? channels.high.N_prop : 0
    return Dict{String,Any}(
        "use_indices" => false,
        "v_low_s" => isdefined(wf, :v_low_s) && !isa(wf.v_low_s, Nothing) ? wf.v_low_s : zeros(ComplexF64, n_low, 0),
        "v_low_p" => isdefined(wf, :v_low_p) && !isa(wf.v_low_p, Nothing) ? wf.v_low_p : zeros(ComplexF64, n_low, 0),
        "v_high_s" => isdefined(wf, :v_high_s) && !isa(wf.v_high_s, Nothing) ? wf.v_high_s : zeros(ComplexF64, n_high, 0),
        "v_high_p" => isdefined(wf, :v_high_p) && !isa(wf.v_high_p, Nothing) ? wf.v_high_p : zeros(ComplexF64, n_high, 0),
    )
end

function build_input_surface_wavefront(modes, side, alpha, wf, shape_ex, shape_ey)
    v_s = isdefined(wf, :v_low_s) && !isa(wf.v_low_s, Nothing) ? wf.v_low_s : zeros(ComplexF64, side.N_prop, 0)
    v_p = isdefined(wf, :v_low_p) && !isa(wf.v_low_p, Nothing) ? wf.v_low_p : zeros(ComplexF64, side.N_prop, 0)
    phase = side.sqrt_nu_prop .* exp.((-1im * DN) .* side.kzdx_prop)
    dz_phase = cos.(side.kzdx_prop ./ 2) .* exp.((-1im * DN) .* side.kzdx_prop) ./ side.sqrt_nu_prop

    B_s_Ex = modes.Ex * (reshape(phase .* alpha.x_s, :, 1) .* v_s)
    B_s_Ey = modes.Ey * (reshape(phase .* alpha.y_s, :, 1) .* v_s)
    B_p_Ex = modes.Ex * (reshape(phase .* alpha.x_p, :, 1) .* v_p)
    B_p_Ey = modes.Ey * (reshape(phase .* alpha.y_p, :, 1) .* v_p)
    B_p_dx = modes.dEz_dx * (reshape(dz_phase .* alpha.z_p, :, 1) .* v_p)
    B_p_dy = modes.dEz_dy * (reshape(dz_phase .* alpha.z_p, :, 1) .* v_p)
    B_Ex = reshape([B_s_Ex B_p_Ex + 1im .* B_p_dx], shape_ex[1], shape_ex[2], 1, :)
    B_Ey = reshape([B_s_Ey B_p_Ey + 1im .* B_p_dy], shape_ey[1], shape_ey[2], 1, :)
    return B_Ex, B_Ey
end

function one_sided_wavefront_surface_fields_manual(syst, wf)
    channels = build_channels_for_system(syst; two_sided=false)
    nx_Ex, ny_Ex, _ = size(syst.epsilon_xx)
    nx_Ey, ny_Ey, _ = size(syst.epsilon_yy)
    modes = prop_modes(channels, channels)
    alpha = prop_alpha(channels)
    B_Ex_low, B_Ey_low = build_input_surface_wavefront(modes, channels, alpha, wf, (nx_Ex, ny_Ex), (nx_Ey, ny_Ey))

    nz_extra_low = 1 + syst.zPML[1].npixels
    padded = deepcopy(syst)
    padded.epsilon_xx = cat(syst.epsilon_low * ones(nx_Ex, ny_Ex, nz_extra_low), syst.epsilon_xx, dims=3)
    padded.epsilon_yy = cat(syst.epsilon_low * ones(nx_Ey, ny_Ey, nz_extra_low), syst.epsilon_yy, dims=3)
    padded.epsilon_zz = cat(syst.epsilon_low * ones(size(syst.epsilon_zz, 1), size(syst.epsilon_zz, 2), nz_extra_low), syst.epsilon_zz, dims=3)
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

    Ex, Ey, Ez, info = mesti(padded, [B_Ex, B_Ey, B_Ez], base_opts())
    Ex = (-2im) .* Ex[:, :, nz_extra_low:end, :]
    Ey = (-2im) .* Ey[:, :, nz_extra_low:end, :]
    Ez = (-2im) .* Ez[:, :, nz_extra_low:end, :]
    return Ex, Ey, Ez, channels, info
end

function main()
    two = make_two_sided_system()

    indexed_surface_Ex, indexed_surface_Ey, indexed_surface_Ez, channels_indexed, indexed_info =
        mesti2s(two, all_channels(), surface_opts(true))
    indexed_Ex, indexed_Ey, indexed_Ez = extend_profile_3d(
        indexed_surface_Ex,
        indexed_surface_Ey,
        indexed_surface_Ez,
        channels_indexed,
        index_input_info(channels_indexed; two_sided=true),
        3,
        2;
        two_sided=true,
    )

    channels_for_wf = build_channels_for_system(two; two_sided=true)
    wf = wavefront()
    wf.v_low_s = wave_matrix(channels_for_wf.low.N_prop, 2; seed=1.0)
    wf.v_low_p = wave_matrix(channels_for_wf.low.N_prop, 1; seed=-0.65)
    wf.v_high_s = wave_matrix(channels_for_wf.high.N_prop, 1; seed=0.35)
    wf.v_high_p = wave_matrix(channels_for_wf.high.N_prop, 2; seed=0.45)
    wf_surface_Ex, wf_surface_Ey, wf_surface_Ez, channels_wf, wf_info =
        mesti2s(two, wf, surface_opts(true))
    wf_Ex, wf_Ey, wf_Ez = extend_profile_3d(
        wf_surface_Ex,
        wf_surface_Ey,
        wf_surface_Ez,
        channels_wf,
        wavefront_input_info(wf, channels_wf; two_sided=true),
        3,
        2;
        two_sided=true,
    )

    one = make_one_sided_system()
    channels_one_for_wf = build_channels_for_system(one; two_sided=false)
    one_wf = wavefront()
    one_wf.v_low_s = wave_matrix(channels_one_for_wf.N_prop, 1; seed=0.80)
    one_wf.v_low_p = wave_matrix(channels_one_for_wf.N_prop, 2; seed=-0.55)
    one_surface_Ex, one_surface_Ey, one_surface_Ez, channels_one, one_info =
        one_sided_wavefront_surface_fields_manual(one, one_wf)
    one_Ex, one_Ey, one_Ez = extend_profile_3d(
        one_surface_Ex,
        one_surface_Ey,
        one_surface_Ez,
        channels_one,
        wavefront_input_info(one_wf, channels_one; two_sided=false),
        2,
        2;
        two_sided=false,
    )

    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_mesti2s_3d_nz_extension_julia_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "solver" => "JULIA",
        "use_single_precision_MUMPS" => false,
        "description" => "V5 corrected-formula Julia fixture for diagonal 3D mesti2s nz_low/nz_high homogeneous field-profile extension.",
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
        "indexed_nz_low" => 3,
        "indexed_nz_high" => 2,
        "indexed_surface_Ex" => indexed_surface_Ex,
        "indexed_surface_Ey" => indexed_surface_Ey,
        "indexed_surface_Ez" => indexed_surface_Ez,
        "indexed_field_Ex" => indexed_Ex,
        "indexed_field_Ey" => indexed_Ey,
        "indexed_field_Ez" => indexed_Ez,
        "indexed_return_field_profile_surface" => indexed_info.opts.return_field_profile,
        "wavefront_nz_low" => 3,
        "wavefront_nz_high" => 2,
        "v_low_s" => wf.v_low_s,
        "v_low_p" => wf.v_low_p,
        "v_high_s" => wf.v_high_s,
        "v_high_p" => wf.v_high_p,
        "wavefront_surface_Ex" => wf_surface_Ex,
        "wavefront_surface_Ey" => wf_surface_Ey,
        "wavefront_surface_Ez" => wf_surface_Ez,
        "wavefront_field_Ex" => wf_Ex,
        "wavefront_field_Ey" => wf_Ey,
        "wavefront_field_Ez" => wf_Ez,
        "wavefront_return_field_profile_surface" => wf_info.opts.return_field_profile,
        "one_epsilon_xx" => one.epsilon_xx,
        "one_epsilon_yy" => one.epsilon_yy,
        "one_epsilon_zz" => one.epsilon_zz,
        "one_epsilon_low" => one.epsilon_low,
        "one_wavelength" => one.wavelength,
        "one_dx" => one.dx,
        "one_xBC" => one.xBC,
        "one_yBC" => one.yBC,
        "one_zPML_npixels" => one.zPML[1].npixels,
        "one_nz_low" => 2,
        "one_nz_high" => 2,
        "one_v_low_s" => one_wf.v_low_s,
        "one_v_low_p" => one_wf.v_low_p,
        "one_surface_Ex" => one_surface_Ex,
        "one_surface_Ey" => one_surface_Ey,
        "one_surface_Ez" => one_surface_Ez,
        "one_field_Ex" => one_Ex,
        "one_field_Ey" => one_Ey,
        "one_field_Ez" => one_Ez,
        "one_return_field_profile_surface" => one_info.opts.return_field_profile,
    )
    payload["kxdx_all"] = channels_indexed.kxdx_all
    payload["kydx_all"] = channels_indexed.kydx_all
    channel_payload!(payload, "low", channels_indexed.low)
    channel_payload!(payload, "high", channels_indexed.high)
    payload["one_kxdx_all"] = channels_one.kxdx_all
    payload["one_kydx_all"] = channels_one.kydx_all
    channel_payload!(payload, "one_low", channels_one)

    matwrite(OUT_PATH, payload)
    println(
        "Wrote ",
        OUT_PATH,
        " with two-sided field shapes Ex=",
        size(wf_Ex),
        " Ey=",
        size(wf_Ey),
        " Ez=",
        size(wf_Ez),
        " and one-sided Ex=",
        size(one_Ex),
    )
end

main()
