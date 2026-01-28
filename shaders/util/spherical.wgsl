fn equal_area_dir_to_square(dir: vec3f) -> vec2f {
    let d = abs(dir);
    let r = sqrt(1.0 - d.z);

    let a = max(d.x, d.y);
    let b = min(d.x, d.y);
    var phi = atan2(b, a) * 2.0 / PI;

    if d.x < d.y {
        phi = 1.0 - phi;
    }

    var p = vec2(r - phi * r, phi * r);

    if dir.z < 0.0 {
        p = 1.0 - p.yx;
    }

    p *= select(sign(dir.xy), vec2(1.0), dir.xy == vec2(0.0));
    return (p + 1.0) * 0.5;
}

fn equal_area_square_to_dir(p: vec2f) -> vec3f {
    let uv = 2 * p - 1;
    let uvp = abs(uv);

    let signed_distance = 1 - (uvp.x + uvp.y);
    let d = abs(signed_distance);
    let r = 1 - d;

    let phi = PI / 4 * select((uvp.y - uvp.x) / r + 1, 1, r == 0);

    let z = copysign(1 - r * r, signed_distance);

    let cos_phi = copysign(cos(phi), uv.x);
    let sin_phi = copysign(sin(phi), uv.y);

    let factor = r * sqrt(2 - r * r);
    return vec3f(cos_phi * factor, sin_phi * factor, z);
}

fn cos_theta(v: vec3f) -> f32 {
    return v.z;
}

fn abs_cos_theta(v: vec3f) -> f32 {
    return abs(v.z);
}

fn cos2_theta(v: vec3f) -> f32 {
    return v.z * v.z;
}

fn sin_theta(v: vec3f) -> f32 {
    return sqrt(sin2_theta(v));
}

fn sin2_theta(v: vec3f) -> f32 {
    return max(0, 1 - cos2_theta(v));
}

fn cos_phi(v: vec3f) -> f32 {
    let sin = sin_theta(v);
    if sin == 0 {
        return 1;
    } else {
        return min(max(-1, v.x / sin), 1);
    }
}

fn sin_phi(v: vec3f) -> f32 {
    let sin = sin_theta(v);
    if sin == 0 {
        return 0;
    } else {
        return min(max(-1, v.y / sin), 1);
    }
}
