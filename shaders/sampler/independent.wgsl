#import /util/misc.wgsl

struct SamplerState {
    state: vec3u,
}

var<private> SAMPLER: SamplerState;

fn sample_init(px: vec2u, sample: u32) {
    SAMPLER.state = hash_3d(vec3(px, sample));
}

fn _sampler_next() -> vec3u {
    SAMPLER.state = hash_3d(SAMPLER.state);
    return SAMPLER.state;
}

fn sample_1d() -> f32 {
    let bits = _sampler_next().x;
    return bits_to_f32(bits);
}

fn sample_2d() -> vec2f {
    let bits = _sampler_next().xy;
    return vec2(bits_to_f32(bits.x), bits_to_f32(bits.y));
}

fn sample_pixel() -> vec2f {
    return sample_2d();
}
