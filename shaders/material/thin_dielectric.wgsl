#import /material.wgsl
#import /texture.wgsl
#import /spectrum.wgsl
#import /util/distr.wgsl
#import /util/misc.wgsl
#import /util/spherical.wgsl
#import dielectric.wgsl

struct ThinDielectricMaterial {
    ior: SpectrumId,
}

fn material_thin_dielectric_evaluate(
    material: ThinDielectricMaterial,
    uv: vec2f,
    wl: Wavelengths
) -> BsdfParams {
    var bsdf: BsdfParams;
    bsdf.id = BSDF_THIN_DIELECTRIC;
    bsdf.v0 = spectrum_sample(material.ior, wl);
    return bsdf;
}

fn bsdf_thin_dielectric_f(bsdf: BsdfParams, wo: vec3f, wi: vec3f) -> vec4f {
    return vec4f();
}

fn bsdf_thin_dielectric_sample(bsdf: BsdfParams, wo: vec3f, random: vec3f) -> BsdfSample {
    var r = fresnel_real(abs_cos_theta(wo), bsdf.v0.x);
    if r < 1 {
        r += (1 - r) * (1 - r) * r / (1 - r * r);
    }

    if random.z < r {
        let wi = vec3f(-wo.xy, wo.z);
        let f = r / abs_cos_theta(wi);
        return BsdfSample(vec4f(f), wi, r, true);
    } else {
        let wi = -wo;
        let f = (1 - r) / abs_cos_theta(wi);
        return BsdfSample(vec4f(f), wi, 1 - r, true);
    }
}

fn bsdf_thin_dielectric_pdf(bsdf: BsdfParams, wo: vec3f, wi: vec3f) -> f32 {
    return 0;
}
