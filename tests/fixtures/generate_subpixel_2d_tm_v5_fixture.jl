"""Generate v5 Julia parity fixtures for 2D TM/TE subpixel smoothing.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_subpixel_2d_tm_v5_fixture.jl'

The fixture pins the smallest Python smoothing scope: 2D TM output and 2D TE
inverse-epsilon outputs for GeometryPrimitives ``Cuboid`` rectangles.  3D
smoothing remains an explicit future slice.
"""

using GeometryPrimitives
using MAT
using MESTI

const OUT_PATH = joinpath(@__DIR__, "subpixel_2d_tm_v5.mat")

function record_rectangular_tm!(payload)
    delta_x = 1.0
    domain = Cuboid([2.0, 2.0], [4.0, 4.0])
    object = Cuboid([1.5, 1.5], [1.5, 1.5])
    domain_epsilon = 1.0
    object_epsilon = 4.0
    yBC = "periodic"
    zBC = "PEC"

    epsilon_xx = mesti_subpixel_smoothing(
        delta_x,
        domain,
        domain_epsilon,
        Shape[object],
        [object_epsilon],
        yBC,
        zBC,
        true,
        false,
        false,
    )
    epsilon_xx_without_sb = mesti_subpixel_smoothing(
        delta_x,
        domain,
        domain_epsilon,
        Shape[object],
        [object_epsilon],
        yBC,
        zBC,
        true,
        false,
        true,
    )
    inv_epsilon_yy, inv_epsilon_zz, inv_epsilon_yz = mesti_subpixel_smoothing(
        delta_x,
        domain,
        domain_epsilon,
        Shape[object],
        [object_epsilon],
        yBC,
        zBC,
        false,
        true,
        false,
    )
    inv_epsilon_yy_without_sb, inv_epsilon_zz_without_sb, inv_epsilon_yz_without_sb = mesti_subpixel_smoothing(
        delta_x,
        domain,
        domain_epsilon,
        Shape[object],
        [object_epsilon],
        yBC,
        zBC,
        false,
        true,
        true,
    )

    payload["rect_delta_x"] = delta_x
    payload["rect_domain_center"] = [2.0, 2.0]
    payload["rect_domain_widths"] = [4.0, 4.0]
    payload["rect_domain_epsilon"] = domain_epsilon
    payload["rect_object_center"] = [1.5, 1.5]
    payload["rect_object_widths"] = [1.5, 1.5]
    payload["rect_object_epsilon"] = object_epsilon
    payload["rect_yBC"] = yBC
    payload["rect_zBC"] = zBC
    payload["rect_epsilon_xx"] = epsilon_xx
    payload["rect_epsilon_xx_without_sb"] = epsilon_xx_without_sb
    payload["rect_inv_epsilon_yy"] = inv_epsilon_yy
    payload["rect_inv_epsilon_zz"] = inv_epsilon_zz
    payload["rect_inv_epsilon_yz"] = inv_epsilon_yz
    payload["rect_inv_epsilon_yy_without_sb"] = inv_epsilon_yy_without_sb
    payload["rect_inv_epsilon_zz_without_sb"] = inv_epsilon_zz_without_sb
    payload["rect_inv_epsilon_yz_without_sb"] = inv_epsilon_yz_without_sb
    println("Recorded rectangular 2D TM smoothing with size ", size(epsilon_xx))
    println("Recorded rectangular 2D TE smoothing with component sizes ", size(inv_epsilon_yy), ", ", size(inv_epsilon_zz), ", ", size(inv_epsilon_yz))
end

function record_periodic_image_tm!(payload)
    delta_x = 1.0
    domain = Cuboid([2.0, 2.0], [4.0, 4.0])
    object = Cuboid([-0.25, 1.5], [1.0, 1.0])
    domain_epsilon = 1.0
    object_epsilon = 3.0
    yBC = "periodic"
    zBC = "periodic"

    epsilon_xx = mesti_subpixel_smoothing(
        delta_x,
        domain,
        domain_epsilon,
        Shape[object],
        [object_epsilon],
        yBC,
        zBC,
        true,
        false,
        false,
    )

    payload["periodic_delta_x"] = delta_x
    payload["periodic_domain_center"] = [2.0, 2.0]
    payload["periodic_domain_widths"] = [4.0, 4.0]
    payload["periodic_domain_epsilon"] = domain_epsilon
    payload["periodic_object_center"] = [-0.25, 1.5]
    payload["periodic_object_widths"] = [1.0, 1.0]
    payload["periodic_object_epsilon"] = object_epsilon
    payload["periodic_yBC"] = yBC
    payload["periodic_zBC"] = zBC
    payload["periodic_epsilon_xx"] = epsilon_xx
    println("Recorded periodic-image 2D TM smoothing with size ", size(epsilon_xx))
end

function main()
    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_subpixel_2d_tm_v5_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "V5 2D TM/TE subpixel smoothing fixture for Cuboid rectangles.",
    )
    record_rectangular_tm!(payload)
    record_periodic_image_tm!(payload)
    matwrite(OUT_PATH, payload)
    println("Wrote ", OUT_PATH)
end

main()
