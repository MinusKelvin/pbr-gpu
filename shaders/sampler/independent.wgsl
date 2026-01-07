#import /util.wgsl

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
    return bitcast<f32>(bits >> 9 | 0x3f800000) - 1;
}

fn sample_2d() -> vec2f {
    let bits = _sampler_next().xy;
    return bitcast<vec2f>(bits >> vec2(9) | vec2(0x3f800000)) - 1;
}

fn sample_pixel() -> vec2f {
    return sample_2d();
}
