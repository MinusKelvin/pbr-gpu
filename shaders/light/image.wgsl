#import /spectrum.wgsl
#import /texture.wgsl
#import /transform.wgsl
#import /util/distr.wgsl

struct ImageLight {
    transform: Transform,
    image: u32,
    scale: f32,
}

fn inf_light_image_emission(light: ImageLight, ray_: Ray, wl: Wavelengths) -> vec4f {
    let ray = transform_ray_inv(light.transform, ray_);
    let uv = equal_area_dir_to_square(normalize(ray.d));
    let texel = vec2u(fract(uv) * vec2f(textureDimensions(IMAGES[light.image])));
    let rgb = textureLoad(IMAGES[light.image], texel, 0).xyz;
    let spectrum = RgbIlluminantSpectrum(rgb, SPECTRUM_D65_1NIT);
    return spectrum_rgb_illuminant_sample(spectrum, wl) * light.scale;
}
