#import /spectrum.wgsl
#import /transform.wgsl
#import /shapes.wgsl

struct AreaLight {
    spectrum: SpectrumId,
    transform_node: NodeId,
    shape: ShapeId,
    two_sided: u32,
}

fn light_area_emission(light: AreaLight, ray: Ray, hit: RaycastResult, wl: Wavelengths) -> vec4f {
    if light.two_sided == 0 && dot(ray.d, hit.n) > 0 {
        return vec4f();
    }
    return spectrum_sample(light.spectrum, wl);
}

fn light_area_sample(light: AreaLight, p: vec3f, wl: Wavelengths, random: vec2f) -> LightSample {
    var tp = p;
    if light.transform_node.id != 0 {
        return LightSample();
    }

    let shape_sample = shape_sample(light.shape, p, random);

    let d = shape_sample.p - p;

    if light.two_sided == 0 && dot(d, shape_sample.ng) > 0 || dot(d, shape_sample.ng) == 0 {
        return LightSample();
    }

    let t_max = length(d);
    let dir = d / t_max;

    let pdf_wrt_solid_angle = shape_sample.pdf_wrt_area
        / abs(dot(-dir, shape_sample.ng))
        * dot(d, d);

    return LightSample(spectrum_sample(light.spectrum, wl), dir, t_max, pdf_wrt_solid_angle);
}
