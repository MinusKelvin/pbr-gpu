#import /spectrum.wgsl
#import /texture.wgsl
#import /transform.wgsl
#import /util/distr.wgsl
#import /util/table_sample.wgsl

struct ImageLight {
    transform: Transform,
    image: u32,
    scale: f32,
    light_sampling_path: u32,
    samp: TableSampler2d
}

fn inf_light_image_emission(light: ImageLight, ray_: Ray, wl: Wavelengths) -> vec4f {
    let ray = transform_ray_inv(light.transform, ray_);
    let uv = equal_area_dir_to_square(normalize(ray.d));
    let texel = vec2u(fract(uv) * vec2f(textureDimensions(IMAGES[light.image])));
    let rgb = textureLoad(IMAGES[light.image], texel, 0).xyz;
    let spectrum = RgbIlluminantSpectrum(rgb, SPECTRUM_D65_1NIT);
    return spectrum_rgb_illuminant_sample(spectrum, wl) * light.scale;
}

fn light_image_sample(light: ImageLight, ref_p: vec3f, wl: Wavelengths, random: vec2f) -> LightSample {
    let uv = table_2d_sample(light.samp, random);
    let dir = transform_vector(light.transform, equal_area_square_to_dir(uv.value));

    let texel = vec2u(fract(uv.value) * vec2f(textureDimensions(IMAGES[light.image])));
    let rgb = textureLoad(IMAGES[light.image], texel, 0).xyz;
    let spectrum = RgbIlluminantSpectrum(rgb, SPECTRUM_D65_1NIT);
    let emission = spectrum_rgb_illuminant_sample(spectrum, wl) * light.scale;

    return LightSample(emission, dir, FLOAT_MAX, uv.pdf / (2 * TWO_PI));
}

fn light_image_pdf(light: ImageLight, ref_p: vec3f, d: vec3f) -> f32 {
    let dir = transform_vector_inv(light.transform, d);
    let uv = equal_area_dir_to_square(normalize(dir));

    return table_2d_pdf(light.samp, uv) / (2 * TWO_PI);
}
