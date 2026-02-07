#import /material.wgsl
#import /texture.wgsl
#import /spectrum.wgsl
#import /util/distr.wgsl
#import /util/misc.wgsl
#import /util/spherical.wgsl
#import trowbridge_reitz.wgsl

struct DielectricMaterial {
    normal_map: u32,
    ior: SpectrumId,
    roughness_u: TextureId,
    roughness_v: TextureId,
}

fn material_dielectric_evaluate(material: DielectricMaterial, uv: vec2f, wl: Wavelengths) -> BsdfParams {
    var bsdf: BsdfParams;
    bsdf.id = BSDF_DIELECTRIC;
    bsdf.v0 = spectrum_sample(material.ior, wl);
    let alpha = trowbridge_reitz_adjust_alpha(vec2f(
        texture_evaluate(material.roughness_u, uv, wl).x,
        texture_evaluate(material.roughness_v, uv, wl).x,
    ));
    bsdf.v1 = vec4f(alpha, 0, 0);
    return bsdf;
}

fn bsdf_dielectric_f(bsdf: BsdfParams, wo: vec3f, wi: vec3f) -> vec4f {
    let alpha = bsdf.v1.xy;
    if bsdf.v0.x == 1 || trowbridge_reitz_is_smooth(alpha) {
        return vec4f();
    }

    let ghv = generalized_half_vector(wo, wi, bsdf.v0.x);
    if ghv.ior == 0 {
        return vec4f();
    }
    let nm = ghv.nm;
    let ior = ghv.ior;

    let r = fresnel_real(dot(wo, nm), bsdf.v0.x);

    if ghv.reflect {
        return vec4f(
            r
            * trowbridge_reitz_ndf(alpha, nm)
            * trowbridge_reitz_masking_shadowing(alpha, wo, wi)
            / abs(4 * cos_theta(wo) * cos_theta(wi))
        );
    } else {
        let d = dot(wi, nm) + dot(wo, nm) / ior;
        let denom = d * d * cos_theta(wi) * cos_theta(wo);
        let f = (1 - r)
            * trowbridge_reitz_ndf(alpha, nm)
            * trowbridge_reitz_masking_shadowing(alpha, wo, wi)
            * abs(dot(wi, nm) * dot(wo, nm) / denom)
            / (ior * ior);
        return vec4f(f);
    }
}

fn bsdf_dielectric_sample(bsdf: BsdfParams, wo: vec3f, random: vec3f) -> BsdfSample {
    let alpha = bsdf.v1.xy;
    if bsdf.v0.x == 1 || trowbridge_reitz_is_smooth(alpha) {
        let r = fresnel_real(cos_theta(wo), bsdf.v0.x);

        if random.z < r {
            let wi = vec3f(-wo.xy, wo.z);
            let f = r / abs_cos_theta(wi);
            return BsdfSample(vec4f(f), wi, r, true);
        } else {
            let refr = refract_sane(wo, vec3f(0, 0, 1), bsdf.v0.x);
            if refr.ior == 0 {
                return BsdfSample();
            }
            let f = (1 - r) / abs_cos_theta(refr.dir) / (refr.ior * refr.ior);
            return BsdfSample(vec4f(f), refr.dir, 1 - r, true);
        }
    }

    let nm = trowbridge_reitz_sample(alpha, wo, random.xy);
    let r = fresnel_real(dot(wo, nm), bsdf.v0.x);

    if random.z < r {
        let wi = -reflect(wo, nm);
        if wi.z * wo.z < 0 {
            return BsdfSample();
        }
        let pdf = trowbridge_reitz_visible_ndf(alpha, wo, nm) / (4 * abs(dot(wo, nm))) * r;
        let f = r
            * trowbridge_reitz_ndf(alpha, nm)
            * trowbridge_reitz_masking_shadowing(alpha, wi, wo)
            / (4 * cos_theta(wi) * cos_theta(wo));
        return BsdfSample(vec4f(f), wi, pdf, false);
    } else {
        let refr = refract_sane(wo, nm, bsdf.v0.x);
        if refr.ior == 0 || refr.dir.z * wo.z > 0 {
            return BsdfSample();
        }

        let d = dot(refr.dir, nm) + dot(wo, nm) / refr.ior;
        let dnm_dwi = abs(dot(refr.dir, nm)) / (d * d);
        let pdf = trowbridge_reitz_visible_ndf(alpha, wo, nm) * dnm_dwi * (1 - r);

        let denom = d * d * cos_theta(refr.dir) * cos_theta(wo);
        let f = (1 - r)
            * trowbridge_reitz_ndf(alpha, nm)
            * trowbridge_reitz_masking_shadowing(alpha, wo, refr.dir)
            * abs(dot(refr.dir, nm) * dot(wo, nm) / denom)
            / (refr.ior * refr.ior);

        return BsdfSample(vec4f(f), refr.dir, pdf, false);
    }
}

fn bsdf_dielectric_pdf(bsdf: BsdfParams, wo: vec3f, wi: vec3f) -> f32 {
    let alpha = bsdf.v1.xy;
    if bsdf.v0.x == 1 || trowbridge_reitz_is_smooth(alpha) {
        return 0;
    }

    let ghv = generalized_half_vector(wo, wi, bsdf.v0.x);
    if ghv.ior == 0 {
        return 0;
    }
    let nm = ghv.nm;
    let ior = ghv.ior;

    let r = fresnel_real(dot(wo, nm), ior);

    if ghv.reflect {
        return trowbridge_reitz_visible_ndf(alpha, wo, nm) / (4 * abs(dot(wo, nm))) * r;
    } else {
        let d = dot(wi, nm) + dot(wo, nm) / ior;
        let dnm_dwi = abs(dot(wi, nm)) / (d * d);
        return trowbridge_reitz_visible_ndf(alpha, wo, nm) * dnm_dwi * (1 - r);
    }
}

struct _Ghv {
    nm: vec3f,
    ior: f32,
    reflect: bool,
}

fn generalized_half_vector(wo: vec3f, wi: vec3f, eta: f32) -> _Ghv {
    var ior = 1.0;
    let reflect = cos_theta(wi) * cos_theta(wo) > 0;
    if !reflect {
        if cos_theta(wo) > 0 {
            ior = eta;
        } else {
            ior = 1 / eta;
        }
    }

    var nm = wi * ior + wo;
    if cos_theta(wi) == 0 || cos_theta(wo) == 0 || all(nm == vec3f()) {
        return _Ghv();
    }
    nm = normalize(nm);
    if nm.z < 0 {
        nm = -nm;
    }

    if dot(wi, nm) * cos_theta(wi) < 0 || dot(wo, nm) * cos_theta(wo) < 0 {
        return _Ghv();
    }

    return _Ghv(nm, ior, reflect);
}

fn fresnel_real(cos_theta_: f32, ior_: f32) -> f32 {
    var cos_theta = cos_theta_;
    var ior = ior_;
    if cos_theta < 0 {
        ior = 1 / ior;
        cos_theta = -cos_theta;
    }

    let sin2 = 1 - cos_theta * cos_theta;
    let sin2_transmit = sin2 / (ior * ior);
    if sin2_transmit >= 1 {
        return 1;
    }
    let cos_transmit = sqrt(1 - sin2_transmit);

    let r_par = (ior * cos_theta - cos_transmit) / (ior * cos_theta + cos_transmit);
    let r_perp = (cos_theta - ior * cos_transmit) / (cos_theta + ior * cos_transmit);
    return (r_par * r_par + r_perp * r_perp) / 2;
}

struct Refracted {
    dir: vec3f,
    ior: f32,
}

fn refract_sane(w_: vec3f, n_: vec3f, ior_: f32) -> Refracted {
    var w = w_;
    var n = n_;
    var ior = ior_;

    var cos_i = dot(w, n);
    if cos_i < 0 {
        ior = 1 / ior;
        cos_i = -cos_i;
        n = -n;
    }

    let sin2 = max(0, 1 - cos_i * cos_i);
    let sin2_t = sin2 / (ior * ior);
    if sin2_t >= 1 {
        return Refracted();
    }
    let cos_t = sqrt(1 - sin2_t);

    return Refracted(
        -w / ior + (cos_i / ior - cos_t) * n,
        ior,
    );
}
