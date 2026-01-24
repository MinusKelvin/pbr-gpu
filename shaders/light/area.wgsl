#import /spectrum.wgsl
#import /transform.wgsl

struct AreaLight {
    rgb: vec3f,
    illuminant: SpectrumId,
    transform_node: NodeId,
    shape: ShapeId,
}

fn light_area_emission(light: AreaLight, hit: RaycastResult, wl: Wavelengths) -> vec4f {
    return spectrum_rgb_illuminant_sample(light.rgb, wl);
}
