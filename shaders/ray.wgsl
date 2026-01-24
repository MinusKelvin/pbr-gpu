#import /material.wgsl

struct Ray {
    o: vec3f,
    d: vec3f,
    time: f32,
}

struct RaycastResult {
    hit: bool,
    p: vec3f,
    n: vec3f,
    t: f32,
    material: MaterialId,
    light: LightId,
    uv: vec2f,
}
