#import /spectrum.wgsl
#import /texture.wgsl
#import material/diffuse.wgsl

@group(0) @binding(96)
var<storage> DIFFUSE_MATERIALS: array<DiffuseMaterial>;

struct MaterialId {
    id: u32,
}

const MATERIAL_TAG_BITS: u32 = 1;
const MATERIAL_TAG_SHIFT: u32 = 32 - MATERIAL_TAG_BITS;
const MATERIAL_IDX_MASK: u32 = (1 << MATERIAL_TAG_SHIFT) - 1;
const MATERIAL_TAG_MASK: u32 = ~MATERIAL_IDX_MASK;

const MATERIAL_DIFFUSE: u32 = 0 << MATERIAL_TAG_SHIFT;

struct Bsdf {
    id: u32,
    v0: vec4f,
    v1: vec4f,
}

const BSDF_DIFFUSE: u32 = 1;

fn material_evaluate(material: MaterialId, uv: vec2f, wl: Wavelengths) -> Bsdf {
    let idx = material.id & MATERIAL_IDX_MASK;
    switch material.id & MATERIAL_TAG_MASK {
        case MATERIAL_DIFFUSE {
            return material_diffuse_evaluate(DIFFUSE_MATERIALS[idx], uv, wl);
        }
        default {
            return Bsdf();
        }
    }
}

fn bsdf_f(bsdf: Bsdf, wo: vec3f, wi: vec3f) -> vec4f {
    switch bsdf.id {
        case BSDF_DIFFUSE {
            return bsdf_diffuse_f(bsdf, wo, wi);
        }
        default {
            return vec4f();
        }
    }
}
