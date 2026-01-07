#import /sampler.wgsl
#import /spectra.wgsl

@group(1) @binding(0)
var xyz_texture: texture_storage_2d<rgba32float, read_write>;

struct Wavelengths {
    l: vec4f,
}

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
    return textureDimensions(xyz_texture);
}

fn film_add_sample(px: vec2u, wl: Wavelengths, radiance: vec4f) {
    // todo: color matching functions
    textureStore(xyz_texture, px, radiance + textureLoad(xyz_texture, px));
}
