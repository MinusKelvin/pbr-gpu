#import /spectrum.wgsl
#import /transform.wgsl
#import /shapes.wgsl

struct AreaLight {
    spectrum: SpectrumId,
    transform_node: NodeId,
    shape: ShapeId,
    two_sided: u32,
    light_sampling_path: u32,
}

fn light_area_emission(light: AreaLight, ray: Ray, hit: RaycastResult, wl: Wavelengths) -> vec4f {
    if light.two_sided == 0 && dot(ray.d, hit.n) > 0 {
        return vec4f();
    }
    return spectrum_sample(light.spectrum, wl);
}

fn light_area_sample(light: AreaLight, ref_p: vec3f, wl: Wavelengths, random: vec2f) -> LightSample {
    if light.transform_node.id != 0 {
        return LightSample();
    }

    let shape_sample = shape_sample(light.shape, ref_p, random);

    let d = shape_sample.p - ref_p;

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

fn light_area_pdf(light: AreaLight, ref_p: vec3f, dir: vec3f) -> f32 {
    if light.transform_node.id != 0 {
        return 0;
    }

    let result = shape_raycast(light.shape, Ray(ref_p, dir, 0), FLOAT_MAX);
    if !result.hit {
        return 0;
    }

    let d = result.p - ref_p;

    if light.two_sided == 0 && dot(d, result.ng) > 0 || dot(d, result.ng) == 0 {
        return 0;
    }

    let pdf_wrt_area = shape_pdf(light.shape, ref_p, result.p);
    let pdf_wrt_solid_angle = pdf_wrt_area
        / abs(dot(-dir, result.ng))
        * dot(d, d);

    return pdf_wrt_solid_angle;
}

