#import /spectrum.wgsl
#import light/uniform.wgsl
#import light/image.wgsl
#import light/area.wgsl

@group(0) @binding(128)
var<storage> INFINITE_LIGHTS: array<LightId>;
@group(0) @binding(129)
var<storage> UNIFORM_LIGHTS: array<UniformLight>;
@group(0) @binding(130)
var<storage> IMAGE_LIGHTS: array<ImageLight>;
@group(0) @binding(131)
var<storage> AREA_LIGHTS: array<AreaLight>;

struct LightId {
    id: u32
}

const LIGHT_TAG_BITS: u32 = 2;
const LIGHT_TAG_SHIFT: u32 = 32 - LIGHT_TAG_BITS;
const LIGHT_IDX_MASK: u32 = (1 << LIGHT_TAG_SHIFT) - 1;
const LIGHT_TAG_MASK: u32 = ~LIGHT_IDX_MASK;

const LIGHT_UNIFORM: u32 = 0 << LIGHT_TAG_SHIFT;
const LIGHT_IMAGE: u32 = 1 << LIGHT_TAG_SHIFT;
const LIGHT_AREA: u32 = 2 << LIGHT_TAG_SHIFT;

struct LightSample {
    emission: vec4f,
    dir: vec3f,
    t_max: f32,
    pdf_wrt_solid_angle: f32,
}

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

fn light_emission(light: LightId, ray: Ray, hit: RaycastResult, wl: Wavelengths) -> vec4f {
    let idx = light.id & LIGHT_IDX_MASK;
    switch light.id & LIGHT_TAG_MASK {
        case LIGHT_AREA {
            return light_area_emission(AREA_LIGHTS[idx], ray, hit, wl);
        }
        default {
            return vec4f();
        }
    }
}

fn light_sample(light: LightId, ref_p: vec3f, wl: Wavelengths, random: vec2f) -> LightSample {
    let idx = light.id & LIGHT_IDX_MASK;
    switch light.id & LIGHT_TAG_MASK {
        case LIGHT_UNIFORM {
            return light_uniform_sample(UNIFORM_LIGHTS[idx], ref_p, wl, random);
        }
        case LIGHT_IMAGE {
            return light_image_sample(IMAGE_LIGHTS[idx], ref_p, wl, random);
        }
        case LIGHT_AREA {
            return light_area_sample(AREA_LIGHTS[idx], ref_p, wl, random);
        }
        default {
            return LightSample();
        }
    }
}

fn light_pdf(light: LightId, ref_p: vec3f, dir: vec3f) -> f32 {
    let idx = light.id & LIGHT_IDX_MASK;
    switch light.id & LIGHT_TAG_MASK {
        case LIGHT_UNIFORM {
            return light_uniform_pdf(UNIFORM_LIGHTS[idx], ref_p, dir);
        }
        case LIGHT_IMAGE {
            return light_image_pdf(IMAGE_LIGHTS[idx], ref_p, dir);
        }
        case LIGHT_AREA {
            return light_area_pdf(AREA_LIGHTS[idx], ref_p, dir);
        }
        default {
            return 0;
        }
    }
}

fn light_sample_path(light: LightId) -> u32 {
    let idx = light.id & LIGHT_IDX_MASK;
    switch light.id & LIGHT_TAG_MASK {
        case LIGHT_UNIFORM {
            return UNIFORM_LIGHTS[idx].light_sampling_path;
        }
        case LIGHT_IMAGE {
            return IMAGE_LIGHTS[idx].light_sampling_path;
        }
        case LIGHT_AREA {
            return AREA_LIGHTS[idx].light_sampling_path;
        }
        default {
            return 0;
        }
    }
}
