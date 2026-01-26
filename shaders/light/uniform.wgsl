#import /spectrum.wgsl
#import /util/distr.wgsl

struct UniformLight {
    spectrum: SpectrumId,
}

fn inf_light_uniform_emission(light: UniformLight, ray: Ray, wl: Wavelengths) -> vec4f {
    return spectrum_sample(light.spectrum, wl);
}

fn light_uniform_sample(light: UniformLight, wl: Wavelengths, random: vec2f) -> LightSample {
    return LightSample(
        spectrum_sample(light.spectrum, wl),
        sample_uniform_sphere(random),
        1.0 / (2 * TWO_PI),
    );
}
