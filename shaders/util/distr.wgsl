fn sample_uniform_disk(random: vec2f) -> vec2f {
    // concentric mapping is superior to polar mapping wrt. low-discrepancy sampling
    // via pbr-book.org
    let u = random * 2 - 1;
    if all(u == vec2f()) {
        return vec2f();
    }

    var theta: f32;
    var r: f32;
    if abs(u.x) > abs(u.y) {
        r = u.x;
        theta = PI_ON_4 * (u.y / u.x);
    } else {
        r = u.y;
        theta = PI_ON_2 - PI_ON_4 * (u.x / u.y);
    }

    return r * vec2f(cos(theta), sin(theta));
}

fn sample_uniform_sphere(random: vec2f) -> vec3f {
    let z = random.x * 2 - 1;
    let r = sqrt(1 - z*z);
    let phi = TWO_PI * random.y;
    return vec3(r * cos(phi), r * sin(phi), z);
}

fn sample_cosine_hemisphere(random: vec2f) -> vec3f {
    let disk = sample_uniform_disk(random.xy);
    let z = sqrt(1 - dot(disk, disk));
    return vec3f(disk, z);
}

fn pdf_cosine_hemisphere(dir: vec3f) -> f32 {
    return max(0, dir.z / PI);
}

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
