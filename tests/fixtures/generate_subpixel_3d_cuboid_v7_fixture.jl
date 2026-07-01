"""Generate v7 Julia parity fixtures for 3D Cuboid subpixel smoothing.

Run from the project root through the WSL Julia environment:

    wsl.exe --user lky -- bash -ic 'cd "/mnt/d/BaiduSyncdisk/Projects/Q project" && julia Simulation/python/tests/fixtures/generate_subpixel_3d_cuboid_v7_fixture.jl'

The fixture pins small 3D rectangular-Cuboid tensor-output slices of
``mesti_subpixel_smoothing``. It includes both a slab-style Cuboid with
face-planar partial voxels and a finite Cuboid that exercises edge/corner
volume-fraction cuts.
"""

using GeometryPrimitives
using MAT
using MESTI

const OUT_PATH = joinpath(@__DIR__, "subpixel_3d_cuboid_v7.mat")
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

function record_rectangular_3d!(payload)
    delta_x = 1.0
    domain = Cuboid([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    object = Cuboid([0.75, 1.0, 1.0], [1.0, 4.0, 4.0])
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

    payload["rect3d_delta_x"] = delta_x
    payload["rect3d_domain_center"] = [1.0, 1.0, 1.0]
    payload["rect3d_domain_widths"] = [2.0, 2.0, 2.0]
    payload["rect3d_domain_epsilon"] = domain_epsilon
    payload["rect3d_object_center"] = [0.75, 1.0, 1.0]
    payload["rect3d_object_widths"] = [1.0, 4.0, 4.0]
    payload["rect3d_object_epsilon"] = object_epsilon
    payload["rect3d_xBC"] = xBC
    payload["rect3d_yBC"] = yBC
    payload["rect3d_zBC"] = zBC

    for (name, component, component_without_sb) in zip(COMPONENT_NAMES, components, components_without_sb)
        payload["rect3d_$(name)"] = component
        payload["rect3d_$(name)_without_sb"] = component_without_sb
    end

    println("Recorded 3D Cuboid smoothing with component size ", size(components[1]))
end

function record_edge_corner_3d!(payload)
    delta_x = 1.0
    domain = Cuboid([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    object = Cuboid([0.75, 0.75, 0.75], [1.0, 1.0, 1.0])
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

    payload["edge3d_delta_x"] = delta_x
    payload["edge3d_domain_center"] = [1.0, 1.0, 1.0]
    payload["edge3d_domain_widths"] = [2.0, 2.0, 2.0]
    payload["edge3d_domain_epsilon"] = domain_epsilon
    payload["edge3d_object_center"] = [0.75, 0.75, 0.75]
    payload["edge3d_object_widths"] = [1.0, 1.0, 1.0]
    payload["edge3d_object_epsilon"] = object_epsilon
    payload["edge3d_xBC"] = xBC
    payload["edge3d_yBC"] = yBC
    payload["edge3d_zBC"] = zBC

    for (name, component, component_without_sb) in zip(COMPONENT_NAMES, components, components_without_sb)
        payload["edge3d_$(name)"] = component
        payload["edge3d_$(name)_without_sb"] = component_without_sb
    end

    println("Recorded 3D finite Cuboid edge/corner smoothing with component size ", size(components[1]))
end

function main()
    payload = Dict{String,Any}(
        "fixture_format" => 1,
        "generator" => "generate_subpixel_3d_cuboid_v7_fixture.jl",
        "julia_mesti_version" => "0.5.1",
        "description" => "V7 3D subpixel smoothing fixtures for face-planar and edge/corner Cuboids.",
    )
    record_rectangular_3d!(payload)
    record_edge_corner_3d!(payload)
    matwrite(OUT_PATH, payload)
    println("Wrote ", OUT_PATH)
end

main()
