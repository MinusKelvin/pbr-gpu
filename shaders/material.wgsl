#import /spectrum.wgsl
#import /texture.wgsl
#import material/diffuse.wgsl
#import material/diffuse_transmit.wgsl
#import material/conductor.wgsl
#import material/dielectric.wgsl
#import material/thin_dielectric.wgsl
#import material/metallic_workflow.wgsl

@group(0) @binding(96)
var<storage> DIFFUSE_MATERIALS: array<DiffuseMaterial>;
@group(0) @binding(97)
var<storage> DIFFUSE_TRANSMIT_MATERIALS: array<DiffuseTransmitMaterial>;
@group(0) @binding(98)
var<storage> CONDUCTOR_MATERIALS: array<ConductorMaterial>;
@group(0) @binding(99)
var<storage> DIELECTRIC_MATERIALS: array<DielectricMaterial>;
@group(0) @binding(100)
var<storage> THIN_DIELECTRIC_MATERIALS: array<ThinDielectricMaterial>;
@group(0) @binding(101)
var<storage> METALLIC_WORKFLOW_MATERIALS: array<MetallicWorkflowMaterial>;
@group(0) @binding(102)
var<storage> MIX_MATERIALS: array<MixMaterial>;

struct MaterialId {
    id: u32,
}

const MATERIAL_TAG_BITS: u32 = 3;
const MATERIAL_TAG_SHIFT: u32 = 32 - MATERIAL_TAG_BITS;
const MATERIAL_IDX_MASK: u32 = (1 << MATERIAL_TAG_SHIFT) - 1;
const MATERIAL_TAG_MASK: u32 = ~MATERIAL_IDX_MASK;

const MATERIAL_DIFFUSE: u32 = 0 << MATERIAL_TAG_SHIFT;
const MATERIAL_DIFFUSE_TRANSMIT: u32 = 1 << MATERIAL_TAG_SHIFT;
const MATERIAL_CONDUCTOR: u32 = 2 << MATERIAL_TAG_SHIFT;
const MATERIAL_DIELECTRIC: u32 = 3 << MATERIAL_TAG_SHIFT;
const MATERIAL_THIN_DIELECTRIC: u32 = 4 << MATERIAL_TAG_SHIFT;
const MATERIAL_METALLIC_WORKFLOW: u32 = 5 << MATERIAL_TAG_SHIFT;
const MATERIAL_MIX: u32 = 6 << MATERIAL_TAG_SHIFT;

struct BsdfParams {
    id: u32,
    v0: vec4f,
    v1: vec4f,
    v2: vec4f,
}

const BSDF_DIFFUSE: u32 = 1;
const BSDF_DIFFUSE_TRANSMIT: u32 = 2;
const BSDF_CONDUCTOR: u32 = 3;
const BSDF_DIELECTRIC: u32 = 4;
const BSDF_THIN_DIELECTRIC: u32 = 5;
const BSDF_METALLIC_WORKFLOW: u32 = 6;

struct BsdfSample {
    f: vec4f,
    dir: vec3f,
    pdf: f32,
    specular: bool,
}

struct Bsdf {
    from_local: mat3x3f,
    params: BsdfParams,
}

struct MixMaterial {
    m1: MaterialId,
    m2: MaterialId,
    amount: TextureId,
}

fn material_evaluate(material_: MaterialId, hit: RaycastResult, wl: Wavelengths) -> Bsdf {
    var material = material_;
    while (material.id & MATERIAL_TAG_MASK) == MATERIAL_MIX {
        let mix = MIX_MATERIALS[material.id & MATERIAL_IDX_MASK];
        let amount = texture_evaluate(mix.amount, hit.uv, wl).x;
        let h = hash_4d(vec4u(bitcast<vec2u>(hit.uv), mix.m1.id, mix.m2.id)).w;
        let u = bits_to_f32(h);
        if u < amount {
            material = mix.m2;
        } else {
            material = mix.m1;
        }
    }

    var tangent = hit.tangent - dot(hit.tangent, hit.n) * hit.n;
    if dot(tangent, tangent) <= 1.0e-9 {
        tangent = any_orthonormal_vector(hit.n);
    } else {
        tangent = normalize(tangent);
    }

    var bsdf: Bsdf;
    bsdf.from_local = mat3x3f(
        tangent,
        cross(hit.n, tangent),
        hit.n
    );

    let idx = material.id & MATERIAL_IDX_MASK;
    switch material.id & MATERIAL_TAG_MASK {
        case MATERIAL_DIFFUSE {
            bsdf.params = material_diffuse_evaluate(DIFFUSE_MATERIALS[idx], hit.uv, wl);
        }
        case MATERIAL_DIFFUSE_TRANSMIT {
            bsdf.params = material_diffuse_transmit_evaluate(DIFFUSE_TRANSMIT_MATERIALS[idx], hit.uv, wl);
        }
        case MATERIAL_CONDUCTOR {
            bsdf.params = material_conductor_evaluate(CONDUCTOR_MATERIALS[idx], hit.uv, wl);
        }
        case MATERIAL_DIELECTRIC {
            bsdf.params = material_dielectric_evaluate(DIELECTRIC_MATERIALS[idx], hit.uv, wl);
        }
        case MATERIAL_THIN_DIELECTRIC {
            bsdf.params = material_thin_dielectric_evaluate(THIN_DIELECTRIC_MATERIALS[idx], hit.uv, wl);
        }
        case MATERIAL_METALLIC_WORKFLOW {
            bsdf.params = material_metallic_workflow_evaluate(METALLIC_WORKFLOW_MATERIALS[idx], hit.uv, wl);
        }
        default {}
    }

    return bsdf;
}

fn bsdf_terminates_secondary_wavelengths(bsdf: Bsdf) -> bool {
    return bsdf.params.id == BSDF_DIELECTRIC && any(bsdf.params.v0 != vec4f(bsdf.params.v0.x))
        || bsdf.params.id == BSDF_THIN_DIELECTRIC && any(bsdf.params.v0 != vec4f(bsdf.params.v0.x));
}

fn bsdf_f(bsdf: Bsdf, wo_: vec3f, wi_: vec3f) -> vec4f {
    let wo = transpose(bsdf.from_local) * wo_;
    let wi = transpose(bsdf.from_local) * wi_;

    switch bsdf.params.id {
        case BSDF_DIFFUSE {
            return bsdf_diffuse_f(bsdf.params, wo, wi);
        }
        case BSDF_DIFFUSE_TRANSMIT {
            return bsdf_diffuse_transmit_f(bsdf.params, wo, wi);
        }
        case BSDF_CONDUCTOR {
            return bsdf_conductor_f(bsdf.params, wo, wi);
        }
        case BSDF_DIELECTRIC {
            return bsdf_dielectric_f(bsdf.params, wo, wi);
        }
        case BSDF_THIN_DIELECTRIC {
            return bsdf_thin_dielectric_f(bsdf.params, wo, wi);
        }
        case BSDF_METALLIC_WORKFLOW {
            return bsdf_metallic_workflow_f(bsdf.params, wo, wi);
        }
        default {
            return vec4f();
        }
    }
}

fn bsdf_sample(bsdf: Bsdf, wo_: vec3f, random: vec3f) -> BsdfSample {
    let wo = transpose(bsdf.from_local) * wo_;

    var sample: BsdfSample;
    switch bsdf.params.id {
        case BSDF_DIFFUSE {
            sample = bsdf_diffuse_sample(bsdf.params, wo, random);
        }
        case BSDF_CONDUCTOR {
            sample = bsdf_conductor_sample(bsdf.params, wo, random);
        }
        case BSDF_DIELECTRIC {
            sample = bsdf_dielectric_sample(bsdf.params, wo, random);
        }
        case BSDF_THIN_DIELECTRIC {
            sample = bsdf_thin_dielectric_sample(bsdf.params, wo, random);
        }
        case BSDF_METALLIC_WORKFLOW {
            sample = bsdf_metallic_workflow_sample(bsdf.params, wo, random);
        }
        default {
            // this is also the BSDF_DIFFUSE_TRANSMIT case
            var dir = sample_cosine_hemisphere(random.xy);
            let pdf = pdf_cosine_hemisphere(dir);
            if random.z < 0.5 {
                dir.z = -dir.z;
            }
            sample = BsdfSample(
                bsdf_f(bsdf, dir, wo),
                dir,
                pdf / 2.0,
                false,
            );
        }
    }

    sample.dir = bsdf.from_local * sample.dir;
    return sample;
}

fn bsdf_pdf(bsdf: Bsdf, wo_: vec3f, wi_: vec3f) -> f32 {
    let wo = transpose(bsdf.from_local) * wo_;
    let wi = transpose(bsdf.from_local) * wi_;

    switch bsdf.params.id {
        case BSDF_DIFFUSE {
            return bsdf_diffuse_pdf(bsdf.params, wo, wi);
        }
        case BSDF_CONDUCTOR {
            return bsdf_conductor_pdf(bsdf.params, wo, wi);
        }
        case BSDF_DIELECTRIC {
            return bsdf_dielectric_pdf(bsdf.params, wo, wi);
        }
        case BSDF_THIN_DIELECTRIC {
            return bsdf_thin_dielectric_pdf(bsdf.params, wo, wi);
        }
        case BSDF_METALLIC_WORKFLOW {
            return bsdf_metallic_workflow_pdf(bsdf.params, wo, wi);
        }
        default {
            // this is also the BSDF_DIFFUSE_TRANSMIT case
            let pdf = pdf_cosine_hemisphere(vec3f(wi.xy, copysign(wi.z, 1)));
            return pdf / 2;
        }
    }
}

fn bsdf_normal(bsdf: Bsdf) -> vec3f {
    return bsdf.from_local[2];
}
