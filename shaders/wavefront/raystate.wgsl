#import /ray.wgsl

@group(2) @binding(0)
var<storage, read_write> RAY_STATES: array<RayState>;

struct RayState {
    ray: Ray,
    wavelengths: Wavelengths,
    radiance: vec4f,
    px: vec2u,
    sampler_state: SamplerState,
}
