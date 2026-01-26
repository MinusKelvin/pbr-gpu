#import /spectrum.wgsl

struct UniformLight {
    spectrum: SpectrumId,
}

fn inf_light_uniform_emission(light: UniformLight, ray: Ray, wl: Wavelengths) -> vec4f {
    return spectrum_sample(light.spectrum, wl);
}
