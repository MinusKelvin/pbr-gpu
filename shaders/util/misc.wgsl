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

fn blackbody(lambda: f32, temperature: f32) -> f32 {
    const C: f32 = 299792458;
    const H: f32 = 6.62606957e-34;
    const K_B: f32 = 1.3806488e-23;
    // smallest lambda can be is 3.6e-7, the fifth power of which is 6e-33,
    // which is still 5 orders of magnitude away from stressing the range of floats.
    let l = lambda * 1e-9;
    let l2 = l * l;
    let l5 = l2 * l2 * l;
    let e = H * C / (l * K_B * temperature);
    let radiance = 2.0 * H * C * C / (l5 * (exp(e) - 1.0));
    // Planck's law gives radiance per meter, but we use nanometers
    return radiance * 1e-9;
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
