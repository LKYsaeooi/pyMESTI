"""Generate a reduced fixture for the packaged 3D open-channel example.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_example_3d_open_channel_through_disorder_v5_fixture.jl'

The full Julia example builds a large random 3D Ball-smoothed tensor disorder
sample, computes a both-polarization low-to-high transmission matrix, extracts
closed/open channels with SVD, and plots field slices. This fixture keeps the
3D s/p channel, SVD, and wavefront field-profile core on a tiny deterministic
diagonal tensor system suitable for Python/SciPy regression tests.
"""

using LinearAlgebra
using MAT
using MESTI

const OUT_PATH = joinpath(@__DIR__, "example_3d_open_channel_through_disorder_v5.mat")

const EPSILON_IMAG_LOSS = 0.05

function patterned_epsilon(nx::Int, ny::Int, nz::Int, base::Float64)
    epsilon = zeros(ComplexF64, nx, ny, nz)
    for iz in 1:nz
        for iy in 1:ny
            for ix in 1:nx
                epsilon[ix, iy, iz] =
                    base +
                    0.041 * ix +
                    0.026 * iy +
                    0.013 * iz +
                    0.007 * sin(1.7 * ix + 0.3 * iy + 0.5 * iz) +
                    1im * EPSILON_IMAG_LOSS
            end
        end
    end
    return epsilon
end

function make_system()
    nx = 3
    ny = 3
    nz = 1
    syst = Syst()
    syst.epsilon_xx = patterned_epsilon(nx, ny, nz, 1.04)
    syst.epsilon_yy = patterned_epsilon(nx, ny, nz, 1.12)
    syst.epsilon_zz = patterned_epsilon(nx, ny, nz + 1, 1.20)
    syst.epsilon_low = 1.0
    syst.epsilon_high = 1.0
    syst.length_unit = "lambda_0"
    syst.wavelength = 2 * pi / 1.8
    syst.dx = 1.0
    syst.xBC = "periodic"
    syst.yBC = "periodic"
    syst.zPML = [PML(16)]
    return syst
end

function base_opts()
    opts = Opts()
    opts.verbal = false
    opts.solver = "JULIA"
    opts.use_L0_threads = false
    opts.use_single_precision_MUMPS = false
    return opts
end

function channel_selector(side::String)
    selector = channel_type()
    selector.side = side
    selector.polarization = "both"
    return selector
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

function main()
    syst = make_system()

    input = channel_selector("low")
    output = channel_selector("high")
    opts_transmission = base_opts()

    t0 = time()
    transmission, channels, info_transmission = mesti2s(syst, input, output, opts_transmission)
    elapsed_transmission = time() - t0

    _, sigma, v = svd(transmission)
    tau = sigma .^ 2
    v_open = v[:, 1]
    v_closed = v[:, end]
    N_prop_low = channels.low.N_prop
    normal_index_julia = round(Int, (N_prop_low + 1) / 2)
    normal_index_zero_based = normal_index_julia - 1
    normal_s_column_julia = normal_index_julia
    normal_p_column_julia = N_prop_low + normal_index_julia
    normal_s_column_zero_based = normal_s_column_julia - 1
    normal_p_column_zero_based = normal_p_column_julia - 1

    T_avg = sum(abs2.(transmission)) / (2 * N_prop_low)
    T_PW_s = sum(abs2.(transmission[:, normal_s_column_julia]))
    T_PW_p = sum(abs2.(transmission[:, normal_p_column_julia]))
    T_closed = sigma[end]^2
    T_open = sigma[1]^2

    input_wavefront = wavefront()
    input_wavefront.v_low_s = zeros(ComplexF64, N_prop_low, 2)
    input_wavefront.v_low_p = zeros(ComplexF64, N_prop_low, 3)
    input_wavefront.v_low_s[:, 1] = v_closed[1:N_prop_low]
    input_wavefront.v_low_p[:, 1] = v_closed[(N_prop_low + 1):(2 * N_prop_low)]
    input_wavefront.v_low_s[:, 2] = v_open[1:N_prop_low]
    input_wavefront.v_low_p[:, 2] = v_open[(N_prop_low + 1):(2 * N_prop_low)]
    input_wavefront.v_low_p[normal_index_julia, 3] = 1

    opts_field = base_opts()
    t0 = time()
    Ex, Ey, Ez, channels_field, info_field = mesti2s(syst, input_wavefront, opts_field)
    elapsed_field = time() - t0

    Ex_closed = Ex[:, :, :, 1] + Ex[:, :, :, 3]
    Ey_closed = Ey[:, :, :, 1] + Ey[:, :, :, 3]
    Ez_closed = Ez[:, :, :, 1] + Ez[:, :, :, 3]
    Ex_open = Ex[:, :, :, 2] + Ex[:, :, :, 4]
    Ey_open = Ey[:, :, :, 2] + Ey[:, :, :, 4]
    Ez_open = Ez[:, :, :, 2] + Ez[:, :, :, 4]
    Ex_normal_p = Ex[:, :, :, 5]
    Ey_normal_p = Ey[:, :, :, 5]
    Ez_normal_p = Ez[:, :, :, 5]
    open_ex_normalization_factor = maximum(abs.(Ex_open))

    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_example_3d_open_channel_through_disorder_v5_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "solver" => "JULIA",
        "description" => "Reduced packaged example fixture for 3D open channel through disorder.",
        "use_single_precision_MUMPS" => false,
        "epsilon_xx" => syst.epsilon_xx,
        "epsilon_yy" => syst.epsilon_yy,
        "epsilon_zz" => syst.epsilon_zz,
        "epsilon_low" => syst.epsilon_low,
        "epsilon_high" => syst.epsilon_high,
        "epsilon_imag_loss" => EPSILON_IMAG_LOSS,
        "wavelength" => syst.wavelength,
        "dx" => syst.dx,
        "xBC" => syst.xBC,
        "yBC" => syst.yBC,
        "zPML_npixels" => syst.zPML[1].npixels,
        "nx_Ex" => size(syst.epsilon_xx, 1),
        "ny_Ex" => size(syst.epsilon_xx, 2),
        "nz_Ex" => size(syst.epsilon_xx, 3),
        "nx_Ey" => size(syst.epsilon_yy, 1),
        "ny_Ey" => size(syst.epsilon_yy, 2),
        "nz_Ey" => size(syst.epsilon_yy, 3),
        "nx_Ez" => size(syst.epsilon_zz, 1),
        "ny_Ez" => size(syst.epsilon_zz, 2),
        "nz_Ez" => size(syst.epsilon_zz, 3),
        "kxdx_all" => channels.kxdx_all,
        "kydx_all" => channels.kydx_all,
        "transmission" => transmission,
        "singular_values" => sigma,
        "transmission_eigenvalues" => tau,
        "open_channel" => v_open,
        "closed_channel" => v_closed,
        "normal_index_julia" => normal_index_julia,
        "normal_index_zero_based" => normal_index_zero_based,
        "normal_s_column_julia" => normal_s_column_julia,
        "normal_s_column_zero_based" => normal_s_column_zero_based,
        "normal_p_column_julia" => normal_p_column_julia,
        "normal_p_column_zero_based" => normal_p_column_zero_based,
        "v_low_s" => input_wavefront.v_low_s,
        "v_low_p" => input_wavefront.v_low_p,
        "T_avg" => T_avg,
        "T_PW" => T_PW_s,
        "T_PW_s" => T_PW_s,
        "T_PW_p" => T_PW_p,
        "T_closed" => T_closed,
        "T_open" => T_open,
        "field_Ex" => Ex,
        "field_Ey" => Ey,
        "field_Ez" => Ez,
        "field_profile_shape_Ex" => collect(size(Ex)),
        "field_profile_shape_Ey" => collect(size(Ey)),
        "field_profile_shape_Ez" => collect(size(Ez)),
        "combined_closed_Ex" => Ex_closed,
        "combined_closed_Ey" => Ey_closed,
        "combined_closed_Ez" => Ez_closed,
        "combined_open_Ex" => Ex_open,
        "combined_open_Ey" => Ey_open,
        "combined_open_Ez" => Ez_open,
        "normal_p_Ex" => Ex_normal_p,
        "normal_p_Ey" => Ey_normal_p,
        "normal_p_Ez" => Ez_normal_p,
        "open_ex_normalization_factor" => open_ex_normalization_factor,
        "normalized_closed_Ex" => Ex_closed ./ open_ex_normalization_factor,
        "normalized_open_Ex" => Ex_open ./ open_ex_normalization_factor,
        "normalized_normal_p_Ex" => Ex_normal_p ./ open_ex_normalization_factor,
        "return_field_profile_transmission" => info_transmission.opts.return_field_profile,
        "return_field_profile_field" => info_field.opts.return_field_profile,
        "elapsed_transmission_seconds" => elapsed_transmission,
        "elapsed_field_seconds" => elapsed_field,
    )
    channel_payload!(payload, "low", channels.low)
    channel_payload!(payload, "high", channels.high)
    channel_payload!(payload, "field_low", channels_field.low)
    channel_payload!(payload, "field_high", channels_field.high)

    matwrite(OUT_PATH, payload)
    println("Recorded reduced 3D open-channel example with Ex epsilon size ", size(syst.epsilon_xx))
    println("N_prop per polarization = ", N_prop_low)
    println("transmission singular values = ", sigma)
    println("T_avg = ", T_avg, ", T_PW_s = ", T_PW_s, ", T_PW_p = ", T_PW_p, ", T_open = ", T_open)
    println("field profile sizes Ex=", size(Ex), " Ey=", size(Ey), " Ez=", size(Ez))
    println("Wrote ", OUT_PATH)
end

main()
