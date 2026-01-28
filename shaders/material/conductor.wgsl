#import /material.wgsl
#import /texture.wgsl
#import /spectrum.wgsl
#import /util/distr.wgsl
#import /util/misc.wgsl
#import /util/spherical.wgsl
#import trowbridge_reitz.wgsl

struct ConductorMaterial {
    ior_re: SpectrumId,
    ior_im: SpectrumId,
    roughness_u: TextureId,
    roughness_v: TextureId,
}

fn material_conductor_evaluate(material: ConductorMaterial, uv: vec2f, wl: Wavelengths) -> Bsdf {
    var bsdf: Bsdf;
    bsdf.id = BSDF_CONDUCTOR;
    bsdf.v0 = spectrum_sample(material.ior_re, wl);
    bsdf.v1 = -spectrum_sample(material.ior_im, wl);
    bsdf.v2.x = texture_evaluate(material.roughness_u, uv, wl).x;
    bsdf.v2.y = texture_evaluate(material.roughness_v, uv, wl).x;
    bsdf.v2 = vec4f(vec2f(length(bsdf.v2.xy)), 0, 0);
    return bsdf;
}

fn bsdf_conductor_f(bsdf: Bsdf, wo: vec3f, wi: vec3f) -> vec4f {
    if wi.z * wo.z < 0 {
        return vec4f();
    }

    let alpha = bsdf.v2.xy;

    if trowbridge_reitz_is_smooth(alpha) {
        return vec4f();
    }

    let cos_theta_o = abs_cos_theta(wo);
    let cos_theta_i = abs_cos_theta(wi);
    if cos_theta_i == 0 || cos_theta_o == 0 {
        return vec4f();
    }

    let nm = normalize(wi + wo);

    let cos_theta = abs(dot(wo, nm));
    let f = vec4f(
        fresnel_complex(cos_theta, vec2f(bsdf.v0.x, bsdf.v1.x)),
        fresnel_complex(cos_theta, vec2f(bsdf.v0.y, bsdf.v1.y)),
        fresnel_complex(cos_theta, vec2f(bsdf.v0.z, bsdf.v1.z)),
        fresnel_complex(cos_theta, vec2f(bsdf.v0.w, bsdf.v1.w)),
    );

    return f
        * trowbridge_reitz_ndf(alpha, nm)
        * trowbridge_reitz_masking_shadowing(alpha, wi, wo)
        / (4 * cos_theta_i * cos_theta_o);
}

fn bsdf_conductor_sample(bsdf: Bsdf, wo: vec3f, random: vec3f) -> BsdfSample {
    let alpha = bsdf.v2.xy;
    if trowbridge_reitz_is_smooth(alpha) {
        let cos_theta = abs_cos_theta(wo);
        let new_dir = vec3f(-wo.xy, wo.z);
        let f = vec4f(
            fresnel_complex(cos_theta, vec2f(bsdf.v0.x, bsdf.v1.x)),
            fresnel_complex(cos_theta, vec2f(bsdf.v0.y, bsdf.v1.y)),
            fresnel_complex(cos_theta, vec2f(bsdf.v0.z, bsdf.v1.z)),
            fresnel_complex(cos_theta, vec2f(bsdf.v0.w, bsdf.v1.w)),
        ) / cos_theta;
        return BsdfSample(f, new_dir, 1, true);
    }

    let nm = trowbridge_reitz_sample(alpha, wo, random.xy);
    let wi = -reflect(wo, nm);
    if wi.z * wo.z < 0 {
        return BsdfSample();
    }
    let pdf = trowbridge_reitz_visible_ndf(alpha, wo, nm)
        / (4 * abs(dot(wo, nm)));

    let cos_theta_o = abs_cos_theta(wo);
    let cos_theta_i = abs_cos_theta(wi);

    let cos_theta = abs(dot(wo, nm));
    let fr = vec4f(
        fresnel_complex(cos_theta, vec2f(bsdf.v0.x, bsdf.v1.x)),
        fresnel_complex(cos_theta, vec2f(bsdf.v0.y, bsdf.v1.y)),
        fresnel_complex(cos_theta, vec2f(bsdf.v0.z, bsdf.v1.z)),
        fresnel_complex(cos_theta, vec2f(bsdf.v0.w, bsdf.v1.w)),
    );

    let f = fr
        * trowbridge_reitz_ndf(alpha, nm)
        * trowbridge_reitz_masking_shadowing(alpha, wi, wo)
        / (4 * cos_theta_i * cos_theta_o);

    return BsdfSample(f, wi, pdf, false);
}

fn fresnel_complex(cos_theta: f32, ior: vec2f) -> f32 {
    let sin2_theta = 1 - cos_theta * cos_theta;
    let sin2_theta_transmit = complex_div(vec2f(sin2_theta, 0), complex_mul(ior, ior));
    let cos_theta_transmit = complex_sqrt(vec2f(1, 0) - sin2_theta_transmit);

    let r_par = complex_div(
        complex_mul(ior, vec2f(cos_theta, 0)) - cos_theta_transmit,
        complex_mul(ior, vec2f(cos_theta, 0)) + cos_theta_transmit,
    );
    let r_perp = complex_div(
        vec2f(cos_theta, 0) - complex_mul(ior, cos_theta_transmit),
        vec2f(cos_theta, 0) + complex_mul(ior, cos_theta_transmit),
    );

    return (dot(r_par, r_par) + dot(r_perp, r_perp)) / 2;
}
