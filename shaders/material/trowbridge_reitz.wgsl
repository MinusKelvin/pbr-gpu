#import /util/spherical.wgsl

fn trowbridge_reitz_is_smooth(alpha: vec2f) -> bool {
    return max(alpha.x, alpha.y) < 1.0e-3;
}

fn trowbridge_reitz_adjust_alpha(alpha_: vec2f) -> vec2f {
    var alpha = alpha_;
    if !trowbridge_reitz_is_smooth(alpha) {
        alpha = max(alpha, vec2f(1.0e-6));
    }
    return alpha;
}

fn trowbridge_reitz_ndf(alpha: vec2f, nm: vec3f) -> f32 {
    if cos2_theta(nm) == 0 {
        return 0;
    }
    let tan2 = sin2_theta(nm) / cos2_theta(nm);
    let cos4 = cos2_theta(nm) * cos2_theta(nm);

    let skew = vec2f(cos_phi(nm), sin_phi(nm)) / alpha;
    let e = 1 + tan2 * dot(skew, skew);
    let denom = PI * alpha.x * alpha.y * cos4 * e * e;

    return 1 / denom;
}

fn trowbridge_reitz_visible_ndf(alpha: vec2f, w: vec3f, nm: vec3f) -> f32 {
    return trowbridge_reitz_masking(alpha, w)
        / abs_cos_theta(w)
        * trowbridge_reitz_ndf(alpha, nm)
        * abs(dot(w, nm));
}

fn trowbridge_reitz_sample(alpha: vec2f, w: vec3f, random: vec2f) -> vec3f {
    var wh = normalize(vec3f(alpha * w.xy, w.z));
    if wh.z < 0 {
        wh = -wh;
    }

    var t1: vec3f;
    if wh.z < 0.99999 {
        t1 = normalize(cross(vec3f(0, 0, 1), wh));
    } else {
        t1 = vec3f(1, 0, 0);
    }
    let t2 = cross(wh, t1);

    var p = sample_uniform_disk(random);
    let h = sqrt(1 - p.x * p.x);
    p.y = mix(h, p.y, (1 + wh.z) / 2);

    let pz = sqrt(max(0, 1 - dot(p, p)));
    let nh = p.x * t1 + p.y * t2 + pz * wh;

    return normalize(vec3f(alpha * nh.xy, max(1.0e-6, nh.z)));
}

fn trowbridge_reitz_lambda(alpha: vec2f, w: vec3f) -> f32 {
    if cos2_theta(w) == 0 {
        return 0;
    }
    let skew = vec2f(cos_phi(w), sin_phi(w)) * alpha;
    let v = dot(skew, skew) * sin2_theta(w) / cos2_theta(w);
    return (sqrt(1 + v) - 1) / 2;
}

fn trowbridge_reitz_masking(alpha: vec2f, w: vec3f) -> f32 {
    return 1 / (1 + trowbridge_reitz_lambda(alpha, w));
}

fn trowbridge_reitz_masking_shadowing(alpha: vec2f, wo: vec3f, wi: vec3f) -> f32 {
    return 1 / (1 + trowbridge_reitz_lambda(alpha, wo) + trowbridge_reitz_lambda(alpha, wi));
}


