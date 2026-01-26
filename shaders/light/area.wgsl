#import /spectrum.wgsl
#import /transform.wgsl

struct AreaLight {
    spectrum: SpectrumId,
    transform_node: NodeId,
    shape: ShapeId,
}

fn light_area_emission(light: AreaLight, hit: RaycastResult, wl: Wavelengths) -> vec4f {
    return spectrum_sample(light.spectrum, wl);
}
