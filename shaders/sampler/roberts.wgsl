#import /util/misc.wgsl

const MAX_DIM: u32 = 256;

@group(1) @binding(8)
var<storage> ROBERTS_ALPHA: array<u32, MAX_DIM>;

struct SamplerState {
    hash_key: vec2u,
    sample: u32,
    dim: u32,
}

var<private> SAMPLER: SamplerState;

fn sample_init(px: vec2u, sample: u32) {
    SAMPLER.hash_key = px;
    SAMPLER.sample = sample;
}

fn sample_1d() -> f32 {
    let dim = SAMPLER.dim;
    SAMPLER.dim++;
    if dim >= MAX_DIM {
        // fallback to independent sampling when max dimension is exceeded
        return bits_to_f32(hash_4d(vec4(SAMPLER.hash_key, SAMPLER.sample, dim)).w);
    }

    let h = hash_3d(vec3(SAMPLER.hash_key, dim));
    var v = h.z + SAMPLER.sample * ROBERTS_ALPHA[dim];

    // scramble to mitigate patterns in low-dimensional projections
    v = fast_owen_scramble(v, h.y);

    // we require a 0.32 fixed-point to float conversion.
    // this is how `bits_to_f32` is implemented, but we don't use it because it may be updated
    // to produce uniform floats in a different way.
    return bitcast<f32>(v >> 9 | 0x3f800000) - 1;
}

fn sample_2d() -> vec2f {
    return vec2f(sample_1d(), sample_1d());
}

fn sample_pixel() -> vec2f {
    return sample_2d();
}
