const FLOAT_MAX = 3.40282347e38;

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
