#import /spectrum.wgsl

struct UniformLight {
    rgb: vec3f,
    illuminant: SpectrumId,
}

fn inf_light_uniform_emission(light: UniformLight, ray: Ray, wl: Wavelengths) -> vec4f {
    return spectrum_rgb_illuminant_sample(light.rgb, wl);
}
