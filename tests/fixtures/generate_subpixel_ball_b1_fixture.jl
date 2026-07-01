"""Generate B1 Julia parity fixtures for Ball subpixel smoothing.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_subpixel_ball_b1_fixture.jl'

The fixture pins a tiny 2D Ball case with TM and TE outputs and a tiny 3D Ball
case with all tensor components. Both use periodic boundaries so returned
arrays keep the full Yee-grid shape.
"""

using GeometryPrimitives
using MAT
using MESTI

const OUT_PATH = joinpath(@__DIR__, "subpixel_ball_b1.mat")
const COMPONENT_NAMES = [
    "epsilon_xx",
    "epsilon_xy",
    "epsilon_xz",
    "epsilon_yx",
    "epsilon_yy",
    "epsilon_yz",
    "epsilon_zx",
    "epsilon_zy",
    "epsilon_zz",
]

function record_ball_2d!(payload)
    delta_x = 1.0
    domain = Cuboid([1.5, 1.5], [3.0, 3.0])
    object = Ball([1.25, 1.25], 0.9)
    domain_epsilon = 1.0
    object_epsilon = 3.5
    yBC = "periodic"
    zBC = "periodic"

    epsilon_xx, inv_epsilon = mesti_subpixel_smoothing(
        delta_x,
        domain,
        domain_epsilon,
        Shape[object],
        [object_epsilon],
        yBC,
        zBC,
        true,
        true,
        false,
    )
    epsilon_xx_without_sb, inv_epsilon_without_sb = mesti_subpixel_smoothing(
        delta_x,
        domain,
        domain_epsilon,
        Shape[object],
        [object_epsilon],
        yBC,
        zBC,
        true,
        true,
        true,
    )

    payload["ball2d_delta_x"] = delta_x
    payload["ball2d_domain_center"] = [1.5, 1.5]
    payload["ball2d_domain_widths"] = [3.0, 3.0]
    payload["ball2d_domain_epsilon"] = domain_epsilon
    payload["ball2d_object_center"] = [1.25, 1.25]
    payload["ball2d_object_radius"] = 0.9
    payload["ball2d_object_epsilon"] = object_epsilon
    payload["ball2d_yBC"] = yBC
    payload["ball2d_zBC"] = zBC
    payload["ball2d_epsilon_xx"] = epsilon_xx
    payload["ball2d_epsilon_xx_without_sb"] = epsilon_xx_without_sb
    for (name, component, component_without_sb) in zip(
        ["inv_epsilon_yy", "inv_epsilon_zz", "inv_epsilon_yz"],
        inv_epsilon,
        inv_epsilon_without_sb,
    )
        payload["ball2d_$(name)"] = component
        payload["ball2d_$(name)_without_sb"] = component_without_sb
    end

    println("Recorded 2D Ball smoothing with size ", size(epsilon_xx))
end

function record_ball_3d!(payload)
    delta_x = 1.0
    domain = Cuboid([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    object = Ball([0.9, 1.0, 1.1], 0.8)
    domain_epsilon = 1.0
    object_epsilon = 4.0
    xBC = "periodic"
    yBC = "periodic"
    zBC = "periodic"

    components = mesti_subpixel_smoothing(
        delta_x,
        domain,
        domain_epsilon,
        Shape[object],
        [object_epsilon],
        xBC,
        yBC,
        zBC,
        false,
    )
    components_without_sb = mesti_subpixel_smoothing(
        delta_x,
        domain,
        domain_epsilon,
        Shape[object],
        [object_epsilon],
        xBC,
        yBC,
        zBC,
        true,
    )

    payload["ball3d_delta_x"] = delta_x
    payload["ball3d_domain_center"] = [1.0, 1.0, 1.0]
    payload["ball3d_domain_widths"] = [2.0, 2.0, 2.0]
    payload["ball3d_domain_epsilon"] = domain_epsilon
    payload["ball3d_object_center"] = [0.9, 1.0, 1.1]
    payload["ball3d_object_radius"] = 0.8
    payload["ball3d_object_epsilon"] = object_epsilon
    payload["ball3d_xBC"] = xBC
    payload["ball3d_yBC"] = yBC
    payload["ball3d_zBC"] = zBC

    for (name, component, component_without_sb) in zip(COMPONENT_NAMES, components, components_without_sb)
        payload["ball3d_$(name)"] = component
        payload["ball3d_$(name)_without_sb"] = component_without_sb
    end

    println("Recorded 3D Ball smoothing with component size ", size(components[1]))
end

function main()
    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_subpixel_ball_b1_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "B1 2D and 3D Ball subpixel smoothing fixtures.",
    )
    record_ball_2d!(payload)
    record_ball_3d!(payload)
    matwrite(OUT_PATH, payload)
    println("Wrote ", OUT_PATH)
end

main()
