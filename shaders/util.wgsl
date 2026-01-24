const FLOAT_MAX = 3.40282347e38;
const EPSILON = 1.19209290e-07;

const TWO_PI = 6.28318548;
const PI = 3.14159274;
const PI_ON_2 = 1.57079637;
const PI_ON_4 = 0.785398185;

fn hash_3d(x: vec3u) -> vec3u {
    // pcg3d (Jarzynski and Olano, 2020)
    var v = x;
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

fn hash_4d(x: vec4u) -> vec4u {
    // pcg4d (Jarzynski and Olano, 2020)
    var v = x;
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v ^= v >> vec4(16u);
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

fn copysign(magnitude: f32, sign: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(magnitude) & 0x7FFFFFFFu | bitcast<u32>(sign) & 0x80000000u);
}

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

fn difference_of_products(a1: f32, a2: f32, b1: f32, b2: f32) -> f32 {
    // todo: improve accuracy
    return a1 * a2 - b1 * b2;
}

fn any_orthonormal_frame(n: vec3f) -> mat3x3f {
    // via "Building an orthonormal basis, revisited" (Duff et al. 2017)
    let sign = copysign(1.0, n.z);
    let a = -1 / (sign + n.z);
    let b = n.x * n.y * a;
    return mat3x3(
        vec3(1 + sign * n.x * n.x * a, sign * b, -sign * n.x),
        vec3(b, sign + n.y * n.y * a, -n.y),
        n
    );
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
