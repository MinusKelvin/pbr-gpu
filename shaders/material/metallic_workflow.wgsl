#import /material.wgsl
#import /texture.wgsl
#import /spectrum.wgsl
#import /util/distr.wgsl
#import /util/misc.wgsl
#import /util/spherical.wgsl
#import trowbridge_reitz.wgsl

struct MetallicWorkflowMaterial {
    base_color: TextureId,
    metallic: TextureId,
    roughness_u: TextureId,
    roughness_v: TextureId,
}

fn material_metallic_workflow_evaluate(
    material: MetallicWorkflowMaterial,
    uv: vec2f,
    wl: Wavelengths
) -> BsdfParams {
    var bsdf: BsdfParams;
    bsdf.id = BSDF_METALLIC_WORKFLOW;
    bsdf.v0 = texture_evaluate(material.base_color, uv, wl);
    bsdf.v1.x = texture_evaluate(material.roughness_u, uv, wl).x;
    bsdf.v1.y = texture_evaluate(material.roughness_v, uv, wl).x;
    bsdf.v1.z = texture_evaluate(material.metallic, uv, wl).x;
    return bsdf;
}

fn bsdf_metallic_workflow_f(bsdf: BsdfParams, wo: vec3f, wi: vec3f) -> vec4f {
    if wi.z * wo.z <= 0 {
        return vec4f();
    }

    let base_color = bsdf.v0;
    let alpha = bsdf.v1.xy;
    let metallic = bsdf.v1.z;
    let nm = normalize(wo + wi);
    let r = fresnel_schlick(base_color, metallic, dot(nm, wo));

    var specular_part = vec4f();
    if !trowbridge_reitz_is_smooth(alpha) {
        specular_part = r
            * trowbridge_reitz_ndf(alpha, nm)
            * trowbridge_reitz_masking_shadowing(alpha, wo, wi)
            / (4 * cos_theta(wi) * cos_theta(wo));
    }

    let diffuse_part = base_color * (1 - r) * (1 - metallic) / PI;

    return specular_part + diffuse_part;
}

fn bsdf_metallic_workflow_sample(bsdf: BsdfParams, wo: vec3f, random: vec3f) -> BsdfSample {
    let alpha = bsdf.v1.xy;

    let pr_diffuse = 0.5 * (1 - bsdf.v1.z);

    var wi: vec3f;
    var pdf_diffuse: f32;
    var pdf_specular: f32;

    if random.z < pr_diffuse {
        wi = sample_cosine_hemisphere(random.xy);
        pdf_diffuse = pdf_cosine_hemisphere(wi);
        wi.z = copysign(wi.z, wo.z);

        if !trowbridge_reitz_is_smooth(alpha) {
            let nm = normalize(wi + wo);
            pdf_specular = trowbridge_reitz_visible_ndf(alpha, wo, nm)
                / (4 * abs(dot(wo, nm)));
        }
    } else if trowbridge_reitz_is_smooth(alpha) {
        wi = vec3f(-wo.xy, wo.z);
        let r = fresnel_schlick(bsdf.v0, bsdf.v1.z, abs_cos_theta(wo));
        let f = r / abs_cos_theta(wi);
        return BsdfSample(f, wi, 1 - pr_diffuse, true);
    } else {
        let nm = trowbridge_reitz_sample(alpha, wo, random.xy);
        wi = -reflect(wo, nm);
        if wi.z * wo.z < 0 {
            return BsdfSample();
        }
        pdf_specular = trowbridge_reitz_visible_ndf(alpha, wo, nm)
            / (4 * abs(dot(wo, nm)));

        pdf_diffuse = pdf_cosine_hemisphere(vec3f(wi.xy, abs(wi.z)));
    }

    let pdf = mix(pdf_specular, pdf_diffuse, pr_diffuse);

    let f = bsdf_metallic_workflow_f(bsdf, wo, wi);
    return BsdfSample(f, wi, pdf, false);
}

fn bsdf_metallic_workflow_pdf(bsdf: BsdfParams, wo_: vec3f, wi_: vec3f) -> f32 {
    if wi_.z * wo_.z <= 0 {
        return 0;
    }
    var wo = wo_;
    var wi = wi_;
    if wo.z < 0 {
        wo.z = -wo.z;
        wi.z = -wi.z;
    }

    let alpha = bsdf.v1.xy;
    let nm = normalize(wi + wo);

    let pr_diffuse = 0.5 * (1 - bsdf.v1.z);
    let pdf_diffuse = pdf_cosine_hemisphere(wi);
    var pdf_specular: f32;
    if !trowbridge_reitz_is_smooth(alpha) {
        pdf_specular = trowbridge_reitz_visible_ndf(alpha, wo, nm)
            / (4 * abs(dot(wo, nm)));
    }

    return mix(pdf_specular, pdf_diffuse, pr_diffuse);
}

fn fresnel_schlick(base_color: vec4f, metallic: f32, cos_theta: f32) -> vec4f {
    let f0 = mix(vec4f(0.04), base_color, metallic);
    let v = 1 - cos_theta;
    let v2 = v * v;
    let v5 = v2 * v2 * v;
    return f0 + (1 - f0) * v5;
}
