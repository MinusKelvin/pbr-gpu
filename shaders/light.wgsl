#import /spectrum.wgsl
#import light/uniform.wgsl
#import light/image.wgsl

@group(0) @binding(128)
var<storage> INFINITE_LIGHTS: array<LightId>;

@group(0) @binding(130)
var<storage> UNIFORM_LIGHTS: array<UniformLight>;
@group(0) @binding(131)
var<storage> IMAGE_LIGHTS: array<ImageLight>;

struct LightId {
    id: u32
}

const LIGHT_TAG_BITS: u32 = 1;
const LIGHT_TAG_SHIFT: u32 = 32 - LIGHT_TAG_BITS;
const LIGHT_IDX_MASK: u32 = (1 << LIGHT_TAG_SHIFT) - 1;
const LIGHT_TAG_MASK: u32 = ~LIGHT_IDX_MASK;

const LIGHT_UNIFORM: u32 = 0 << LIGHT_TAG_SHIFT;
const LIGHT_IMAGE: u32 = 1 << LIGHT_TAG_SHIFT;

fn inf_light_emission(light: LightId, ray: Ray, wl: Wavelengths) -> vec4f {
    let idx = light.id & LIGHT_IDX_MASK;
    switch light.id & LIGHT_TAG_MASK {
        case LIGHT_UNIFORM {
            return inf_light_uniform_emission(UNIFORM_LIGHTS[idx], ray, wl);
        }
        case LIGHT_IMAGE {
            return inf_light_image_emission(IMAGE_LIGHTS[idx], ray, wl);
        }
        default {
            return vec4f();
        }
    }
}
