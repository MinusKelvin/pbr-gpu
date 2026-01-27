#import /material.wgsl
#import /texture.wgsl
#import /spectrum.wgsl
#import /util/distr.wgsl
#import /util/misc.wgsl

struct ConductorMaterial {
    ior_re: SpectrumId,
    ior_im: SpectrumId,
    roughness: TextureId,
}

fn material_conductor_evaluate(material: ConductorMaterial, uv: vec2f, wl: Wavelengths) -> Bsdf {
    var bsdf: Bsdf;
    bsdf.id = BSDF_CONDUCTOR;
    bsdf.v0 = spectrum_sample(material.ior_re, wl);
    bsdf.v1 = -spectrum_sample(material.ior_im, wl);
    bsdf.v2.x = texture_evaluate(material.roughness, uv, wl).x;
    return bsdf;
}

fn bsdf_conductor_f(bsdf: Bsdf, wo: vec3f, wi: vec3f) -> vec4f {
    // todo microfacet model
    return vec4f();
}

fn bsdf_conductor_sample(bsdf: Bsdf, wo: vec3f, random: vec3f) -> BsdfSample {
    // todo microfacet model
    let cos_theta = abs(wo.z);
    let new_dir = vec3f(-wo.xy, wo.z);
    let f = vec4f(
        fresnel_complex(cos_theta, vec2f(bsdf.v0.x, bsdf.v1.x)),
        fresnel_complex(cos_theta, vec2f(bsdf.v0.y, bsdf.v1.y)),
        fresnel_complex(cos_theta, vec2f(bsdf.v0.z, bsdf.v1.z)),
        fresnel_complex(cos_theta, vec2f(bsdf.v0.w, bsdf.v1.w)),
    ) / cos_theta;
    return BsdfSample(f, new_dir, 1, true);
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
