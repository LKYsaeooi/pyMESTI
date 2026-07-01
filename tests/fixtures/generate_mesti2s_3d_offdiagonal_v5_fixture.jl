"""Generate v5 Julia parity fixtures for off-diagonal 3D ``mesti2s``.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_mesti2s_3d_offdiagonal_v5_fixture.jl'

The bundle locks down a small Hermitian tensor system with all six
off-diagonal permittivity components present. The two-sided case uses the
public Julia ``mesti2s`` wrapper because Step 6 verifies that high-level
source/projection and zero-padding path directly. The one-sided case uses the
manual channel/direct-``mesti`` strategy because Julia MESTI 0.5.1's high-level
one-sided 3D ``mesti2s`` path references ``N_prop_high`` before it exists; the
manual padding follows Julia's one-sided source branch, including its
``epsilon_low`` fill for off-diagonal homogeneous slabs.
"""

using LinearAlgebra
using MAT
using MESTI

const OUT_PATH = joinpath(@__DIR__, "mesti2s_3d_offdiagonal_v5.mat")

function patterned_real(nx, ny, nz, base)
    epsilon = zeros(Float64, nx, ny, nz)
    for iz in 1:nz
        for iy in 1:ny
            for ix in 1:nx
                epsilon[ix, iy, iz] =
                    base +
                    0.021 * ix +
                    0.013 * iy +
                    0.009 * iz +
                    0.004 * sin(ix + 2 * iy + 3 * iz)
            end
        end
    end
    return epsilon
end

function patterned_complex(nx, ny, nz, scale)
    epsilon = zeros(ComplexF64, nx, ny, nz)
    for iz in 1:nz
        for iy in 1:ny
            for ix in 1:nx
                epsilon[ix, iy, iz] =
                    scale * (0.31 + 0.07 * ix - 0.04 * iy + 0.03 * iz) +
                    im * scale * (0.19 - 0.05 * ix + 0.02 * iy + 0.04 * iz)
            end
        end
    end
    return epsilon
end

function make_hermitian_system()
    syst = Syst()
    syst.epsilon_xx = patterned_real(3, 3, 1, 1.18)
    syst.epsilon_yy = patterned_real(3, 3, 1, 1.31)
    syst.epsilon_zz = patterned_real(3, 3, 2, 1.44)
    syst.epsilon_xy = patterned_complex(3, 3, 1, 0.045)
    syst.epsilon_yx = conj(syst.epsilon_xy)
    syst.epsilon_xz = patterned_complex(3, 3, 1, -0.035)
    syst.epsilon_zx = conj(syst.epsilon_xz)
    syst.epsilon_yz = patterned_complex(3, 3, 1, 0.030)
    syst.epsilon_zy = conj(syst.epsilon_yz)
    syst.epsilon_low = 1.0
    syst.epsilon_high = 1.0
    syst.wavelength = 5.0
    syst.dx = 1.0
    syst.xBC = "periodic"
    syst.yBC = "periodic"
    syst.zPML = [PML(16)]
    return syst
end

function make_one_sided_hermitian_system()
    syst = make_hermitian_system()
    syst.epsilon_high = nothing
    return syst
end

function base_opts()
    opts = Opts()
    opts.verbal = false
    opts.solver = "JULIA"
    opts.use_single_precision_MUMPS = false
    return opts
end

function all_channels()
    spec = channel_type()
    spec.side = "both"
    spec.polarization = "both"
    return spec
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

function one_sided_low_reflection_manual(syst)
    k0dx = (2 * pi / syst.wavelength) * syst.dx
    nx_Ex, ny_Ex, _ = size(syst.epsilon_xx)
    nx_Ey, ny_Ey, _ = size(syst.epsilon_yy)
    nx_Ez, ny_Ez, _ = size(syst.epsilon_zz)
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
    padded.epsilon_xy = cat(syst.epsilon_low * ones(nx_Ez, ny_Ez, nz_extra_low), syst.epsilon_xy, dims=3)
    padded.epsilon_xz = cat(syst.epsilon_low * ones(nx_Ey, ny_Ex, nz_extra_low), syst.epsilon_xz, dims=3)
    padded.epsilon_yx = cat(syst.epsilon_low * ones(nx_Ez, ny_Ez, nz_extra_low), syst.epsilon_yx, dims=3)
    padded.epsilon_yz = cat(syst.epsilon_low * ones(nx_Ey, ny_Ex, nz_extra_low), syst.epsilon_yz, dims=3)
    padded.epsilon_zx = cat(syst.epsilon_low * ones(nx_Ey, ny_Ez, nz_extra_low), syst.epsilon_zx, dims=3)
    padded.epsilon_zy = cat(syst.epsilon_low * ones(nx_Ez, ny_Ex, nz_extra_low), syst.epsilon_zy, dims=3)
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

    S, info = mesti(padded, [B_Ex, B_Ey, B_Ez], [C_Ex, C_Ey, C_Ez], nothing, base_opts())
    S = (-2im) * S
    prefactor = [exp.((-1im * dn) * channels.kzdx_prop); exp.((-1im * dn) * channels.kzdx_prop)]
    S = prefactor .* S .* reshape(prefactor, 1, :)
    D = Diagonal([exp.((-1im * 2 * dn) * channels.kzdx_prop); exp.((-1im * 2 * dn) * channels.kzdx_prop)])
    S = S - D
    return S, channels, info, nz_extra_low
end

function record_case!(payload, prefix, syst)
    both = all_channels()
    S, channels, info = mesti2s(syst, both, both, base_opts())
    residual = maximum(abs.((S' * S) - I(size(S, 1))))
    singular_values = svd(S).S

    payload["$(prefix)_epsilon_xx"] = syst.epsilon_xx
    payload["$(prefix)_epsilon_xy"] = syst.epsilon_xy
    payload["$(prefix)_epsilon_xz"] = syst.epsilon_xz
    payload["$(prefix)_epsilon_yx"] = syst.epsilon_yx
    payload["$(prefix)_epsilon_yy"] = syst.epsilon_yy
    payload["$(prefix)_epsilon_yz"] = syst.epsilon_yz
    payload["$(prefix)_epsilon_zx"] = syst.epsilon_zx
    payload["$(prefix)_epsilon_zy"] = syst.epsilon_zy
    payload["$(prefix)_epsilon_zz"] = syst.epsilon_zz
    payload["$(prefix)_epsilon_low"] = syst.epsilon_low
    payload["$(prefix)_epsilon_high"] = syst.epsilon_high
    payload["$(prefix)_wavelength"] = syst.wavelength
    payload["$(prefix)_dx"] = syst.dx
    payload["$(prefix)_xBC"] = syst.xBC
    payload["$(prefix)_yBC"] = syst.yBC
    payload["$(prefix)_zPML_npixels"] = syst.zPML[1].npixels
    payload["$(prefix)_kxdx_all"] = channels.kxdx_all
    payload["$(prefix)_kydx_all"] = channels.kydx_all
    payload["$(prefix)_S_both"] = S
    payload["$(prefix)_S_both_singular_values"] = singular_values
    payload["$(prefix)_unitarity_residual"] = residual
    payload["$(prefix)_singular_value_max_deviation"] = maximum(abs.(singular_values .- 1))
    payload["$(prefix)_return_field_profile_S"] = info.opts.return_field_profile
    channel_payload!(payload, "$(prefix)_low", channels.low)
    channel_payload!(payload, "$(prefix)_high", channels.high)
    println(
        "Recorded ",
        prefix,
        " with N_prop=(",
        channels.low.N_prop,
        ", ",
        channels.high.N_prop,
        "), S size ",
        size(S),
        ", unitarity residual ",
        residual,
    )
end

function record_one_sided_case!(payload, prefix, syst)
    S, channels, info, nz_extra_low = one_sided_low_reflection_manual(syst)

    payload["$(prefix)_epsilon_xx"] = syst.epsilon_xx
    payload["$(prefix)_epsilon_xy"] = syst.epsilon_xy
    payload["$(prefix)_epsilon_xz"] = syst.epsilon_xz
    payload["$(prefix)_epsilon_yx"] = syst.epsilon_yx
    payload["$(prefix)_epsilon_yy"] = syst.epsilon_yy
    payload["$(prefix)_epsilon_yz"] = syst.epsilon_yz
    payload["$(prefix)_epsilon_zx"] = syst.epsilon_zx
    payload["$(prefix)_epsilon_zy"] = syst.epsilon_zy
    payload["$(prefix)_epsilon_zz"] = syst.epsilon_zz
    payload["$(prefix)_epsilon_low"] = syst.epsilon_low
    payload["$(prefix)_wavelength"] = syst.wavelength
    payload["$(prefix)_dx"] = syst.dx
    payload["$(prefix)_xBC"] = syst.xBC
    payload["$(prefix)_yBC"] = syst.yBC
    payload["$(prefix)_zPML_npixels"] = syst.zPML[1].npixels
    payload["$(prefix)_kxdx_all"] = channels.kxdx_all
    payload["$(prefix)_kydx_all"] = channels.kydx_all
    payload["$(prefix)_S_low"] = S
    payload["$(prefix)_S_low_singular_values"] = svd(S).S
    payload["$(prefix)_return_field_profile_S"] = info.opts.return_field_profile
    payload["$(prefix)_nz_extra_low"] = nz_extra_low
    payload["$(prefix)_offdiagonal_low_padding_value"] = syst.epsilon_low
    channel_payload!(payload, "$(prefix)_low", channels)
    println(
        "Recorded ",
        prefix,
        " with N_prop=",
        channels.N_prop,
        ", S size ",
        size(S),
        ", low off-diagonal padding value ",
        syst.epsilon_low,
    )
end

function main()
    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_mesti2s_3d_offdiagonal_v5_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "solver" => "JULIA",
        "use_single_precision_MUMPS" => false,
        "description" => "V5 Hermitian off-diagonal 3D mesti2s fixture for two-sided s/p scattering and manual one-sided low reflection.",
    )

    record_case!(payload, "hermitian", make_hermitian_system())
    record_one_sided_case!(payload, "one", make_one_sided_hermitian_system())

    matwrite(OUT_PATH, payload)
    println("Wrote ", OUT_PATH)
end

main()
