#import /spectrum.wgsl
#import /util/distr.wgsl

struct DistantLight {
    dir: vec3f,
    cos_radius: f32,
    spectrum: SpectrumId,
    light_sampling_path: u32,
}

fn inf_light_distant_emission(light: DistantLight, ray: Ray, wl: Wavelengths) -> vec4f {
    if dot(ray.d, light.dir) >= light.cos_radius {
        return spectrum_sample(light.spectrum, wl);
    }
    return vec4f();
}

fn light_distant_sample(light: DistantLight, p: vec3f, wl: Wavelengths, random: vec2f) -> LightSample {
    let z = mix(light.cos_radius, 1.0, random.x);
    let r = sqrt(1 - z*z);
    let phi = TWO_PI * random.y;
    let d = vec3(r * cos(phi), r * sin(phi), z);
    let dir = any_orthonormal_frame(light.dir) * d;

    return LightSample(
        spectrum_sample(light.spectrum, wl),
        dir,
        FLOAT_MAX,
        1 / (TWO_PI * (1 - light.cos_radius)),
    );
}

fn light_distant_pdf(light: DistantLight, ref_p: vec3f, dir: vec3f) -> f32 {
    if dot(dir, light.dir) >= light.cos_radius {
        return 1 / (TWO_PI * (1 - light.cos_radius));
    }
    return 0;
}
