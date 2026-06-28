#import /ray.wgsl

@group(2) @binding(0)
var<storage, read_write> RAY_STATES: array<RayState>;
@group(2) @binding(1)
var<storage, read_write> PATH_STATES: array<PathState>;

struct RayState {
    ray: Ray,
    wavelengths: Wavelengths,
    radiance: vec4f,
    px: vec2u,
    sampler_state: SamplerState,
}

struct PathState {
    throughput: vec4f,
    depth: u32,
    bsdf_pdf: f32,
    specular_bounce: u32,
    secondary_terminated: u32,
}
