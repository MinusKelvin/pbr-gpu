#import /spectrum.wgsl
#import /util/distr.wgsl

struct UniformLight {
    spectrum: SpectrumId,
    light_sampling_path: u32,
}

fn inf_light_uniform_emission(light: UniformLight, ray: Ray, wl: Wavelengths) -> vec4f {
    return spectrum_sample(light.spectrum, wl);
}

fn light_uniform_sample(light: UniformLight, p: vec3f, wl: Wavelengths, random: vec2f) -> LightSample {
    return LightSample(
        spectrum_sample(light.spectrum, wl),
        sample_uniform_sphere(random),
        FLOAT_MAX,
        1.0 / (2 * TWO_PI),
    );
}

fn light_uniform_pdf(light: UniformLight, ref_p: vec3f, dir: vec3f) -> f32 {
    return 1 / (2 * TWO_PI);
}
