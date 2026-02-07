#import /material.wgsl
#import /texture.wgsl
#import /spectrum.wgsl
#import /util/distr.wgsl

struct DiffuseMaterial {
    normal_map: u32,
    texture: TextureId
}

fn material_diffuse_evaluate(material: DiffuseMaterial, uv: vec2f, wl: Wavelengths) -> BsdfParams {
    var bsdf: BsdfParams;
    bsdf.id = BSDF_DIFFUSE;
    bsdf.v0 = texture_evaluate(material.texture, uv, wl);
    return bsdf;
}

fn bsdf_diffuse_f(bsdf: BsdfParams, wo: vec3f, wi: vec3f) -> vec4f {
    return vec4f(wo.z * wi.z > 0) * bsdf.v0 / PI;
}

fn bsdf_diffuse_sample(bsdf: BsdfParams, wo: vec3f, random: vec3f) -> BsdfSample {
    let dir = sample_cosine_hemisphere(random.xy);
    let pdf = pdf_cosine_hemisphere(dir);
    return BsdfSample(bsdf.v0 / PI, vec3f(dir.xy, copysign(dir.z, wo.z)), pdf, false);
}

fn bsdf_diffuse_pdf(bsdf: BsdfParams, wo_: vec3f, wi_: vec3f) -> f32 {
    var wo = wo_;
    var wi = wi_;
    if wo.z < 0 {
        wo.z = -wo.z;
        wi.z = -wi.z;
    }
    return pdf_cosine_hemisphere(wi);
}
