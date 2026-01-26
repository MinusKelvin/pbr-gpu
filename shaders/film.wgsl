#import /sampler.wgsl
#import /spectrum.wgsl

@group(1) @binding(0)
var mean_texture: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(1)
var variance_texture: texture_storage_2d<rgba32float, read_write>;

fn film_wavelengths_sample() -> Wavelengths {
    let first = sample_1d();
    let stratified = vec4f(first, first + 0.25, first + 0.5, first + 0.75) % 1;
    let lambda = stratified * (WAVELENGTH_MAX - WAVELENGTH_MIN) + WAVELENGTH_MIN;
    return Wavelengths(lambda);
}

fn film_wavelengths_pdf(wl: Wavelengths) -> vec4f {
    return vec4(1 / (WAVELENGTH_MAX - WAVELENGTH_MIN));
}

fn film_size() -> vec2u {
    return textureDimensions(mean_texture);
}

fn film_add_sample(px: vec2u, wl: Wavelengths, radiance: vec4f) {
    let old = textureLoad(mean_texture, px);
    var s = textureLoad(variance_texture, px).xyz;
    var mean = old.xyz;
    let samples = old.w + 1;

    let x = vec3f(
        dot(spectrum_sample(SPECTRUM_CIE_X, wl) * radiance, vec4f(0.25)),
        dot(spectrum_sample(SPECTRUM_CIE_Y, wl) * radiance, vec4f(0.25)),
        dot(spectrum_sample(SPECTRUM_CIE_Z, wl) * radiance, vec4f(0.25)),
    );

    let delta = x - mean;
    mean += delta / samples;
    let delta2 = x - mean;
    s += delta * delta2;

    textureStore(mean_texture, px, vec4f(mean, samples));
    textureStore(variance_texture, px, vec4f(s, 0));
}
