#import light_sampler/uniform.wgsl

@group(0) @binding(224)
var<storage> ROOT_LS: LightSamplerId;
@group(0) @binding(225)
var<storage> UNIFORM_LIGHT_SAMPLERS: array<UniformLightSampler>;

struct LightSamplerId {
    id: u32,
}

const LIGHT_SAMPLER_TAG_BITS: u32 = 1;
const LIGHT_SAMPLER_TAG_SHIFT: u32 = 32 - LIGHT_SAMPLER_TAG_BITS;
const LIGHT_SAMPLER_IDX_MASK: u32 = (1 << LIGHT_SAMPLER_TAG_SHIFT) - 1;
const LIGHT_SAMPLER_TAG_MASK: u32 = ~LIGHT_SAMPLER_IDX_MASK;

const LIGHT_SAMPLER_UNIFORM: u32 = 0 << LIGHT_SAMPLER_TAG_SHIFT;

struct LightIdSample {
    light: LightId,
    pmf: f32,
}

fn light_sampler_sample(
    ls: LightSamplerId,
    ref_p: vec3f,
    random: f32
) -> LightIdSample {
    let idx = ls.id & LIGHT_SAMPLER_IDX_MASK;
    switch ls.id & LIGHT_SAMPLER_TAG_MASK {
        case LIGHT_SAMPLER_UNIFORM {
            return light_sampler_uniform_sample(UNIFORM_LIGHT_SAMPLERS[idx], ref_p, random);
        }
        default {
            return LightIdSample();
        }
    }
}

fn light_sampler_pmf(ls: LightSamplerId, ref_p: vec3f, light: LightId) -> f32 {
    let idx = ls.id & LIGHT_SAMPLER_IDX_MASK;
    switch ls.id & LIGHT_SAMPLER_TAG_MASK {
        case LIGHT_SAMPLER_UNIFORM {
            return light_sampler_uniform_pmf(UNIFORM_LIGHT_SAMPLERS[idx], ref_p, light);
        }
        default {
            return 0;
        }
    }
}
