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
