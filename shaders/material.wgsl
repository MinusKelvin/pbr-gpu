#import /spectrum.wgsl
#import /texture.wgsl
#import material/diffuse.wgsl
#import material/diffuse_transmit.wgsl
#import material/conductor.wgsl
#import material/dielectric.wgsl

@group(0) @binding(96)
var<storage> DIFFUSE_MATERIALS: array<DiffuseMaterial>;
@group(0) @binding(97)
var<storage> DIFFUSE_TRANSMIT_MATERIALS: array<DiffuseTransmitMaterial>;
@group(0) @binding(98)
var<storage> CONDUCTOR_MATERIALS: array<ConductorMaterial>;
@group(0) @binding(99)
var<storage> DIELECTRIC_MATERIALS: array<DielectricMaterial>;

struct MaterialId {
    id: u32,
}

const MATERIAL_TAG_BITS: u32 = 2;
const MATERIAL_TAG_SHIFT: u32 = 32 - MATERIAL_TAG_BITS;
const MATERIAL_IDX_MASK: u32 = (1 << MATERIAL_TAG_SHIFT) - 1;
const MATERIAL_TAG_MASK: u32 = ~MATERIAL_IDX_MASK;

const MATERIAL_DIFFUSE: u32 = 0 << MATERIAL_TAG_SHIFT;
const MATERIAL_DIFFUSE_TRANSMIT: u32 = 1 << MATERIAL_TAG_SHIFT;
const MATERIAL_CONDUCTOR: u32 = 2 << MATERIAL_TAG_SHIFT;
const MATERIAL_DIELECTRIC: u32 = 3 << MATERIAL_TAG_SHIFT;

struct Bsdf {
    id: u32,
    v0: vec4f,
    v1: vec4f,
    v2: vec4f,
}

const BSDF_DIFFUSE: u32 = 1;
const BSDF_DIFFUSE_TRANSMIT: u32 = 2;
const BSDF_CONDUCTOR: u32 = 3;
const BSDF_DIELECTRIC: u32 = 4;

struct BsdfSample {
    f: vec4f,
    dir: vec3f,
    pdf: f32,
    specular: bool,
}

fn material_evaluate(material: MaterialId, uv: vec2f, wl: Wavelengths) -> Bsdf {
    let idx = material.id & MATERIAL_IDX_MASK;
    switch material.id & MATERIAL_TAG_MASK {
        case MATERIAL_DIFFUSE {
            return material_diffuse_evaluate(DIFFUSE_MATERIALS[idx], uv, wl);
        }
        case MATERIAL_DIFFUSE_TRANSMIT {
            return material_diffuse_transmit_evaluate(DIFFUSE_TRANSMIT_MATERIALS[idx], uv, wl);
        }
        case MATERIAL_CONDUCTOR {
            return material_conductor_evaluate(CONDUCTOR_MATERIALS[idx], uv, wl);
        }
        case MATERIAL_DIELECTRIC {
            return material_dielectric_evaluate(DIELECTRIC_MATERIALS[idx], uv, wl);
        }
        default {
            return Bsdf();
        }
    }
}

fn bsdf_terminates_secondary_wavelengths(bsdf: Bsdf) -> bool {
    return bsdf.id == BSDF_DIELECTRIC && any(bsdf.v0 != vec4f(bsdf.v0.x));
}

fn bsdf_f(bsdf: Bsdf, wo: vec3f, wi: vec3f) -> vec4f {
    switch bsdf.id {
        case BSDF_DIFFUSE {
            return bsdf_diffuse_f(bsdf, wo, wi);
        }
        case BSDF_DIFFUSE_TRANSMIT {
            return bsdf_diffuse_transmit_f(bsdf, wo, wi);
        }
        case BSDF_CONDUCTOR {
            return bsdf_conductor_f(bsdf, wo, wi);
        }
        case BSDF_DIELECTRIC {
            return bsdf_dielectric_f(bsdf, wo, wi);
        }
        default {
            return vec4f();
        }
    }
}

fn bsdf_sample(bsdf: Bsdf, wi: vec3f, random: vec3f) -> BsdfSample {
    switch bsdf.id {
        case BSDF_DIFFUSE {
            return bsdf_diffuse_sample(bsdf, wi, random);
        }
        case BSDF_CONDUCTOR {
            return bsdf_conductor_sample(bsdf, wi, random);
        }
        case BSDF_DIELECTRIC {
            return bsdf_dielectric_sample(bsdf, wi, random);
        }
        default {
            // this is also the BSDF_DIFFUSE_TRANSMIT case
            var dir = sample_cosine_hemisphere(random.xy);
            let pdf = pdf_cosine_hemisphere(dir);
            if random.z < 0.5 {
                dir.z = -dir.z;
            }
            return BsdfSample(
                bsdf_f(bsdf, dir, wi),
                dir,
                pdf / 2.0,
                false,
            );
        }
    }
}
