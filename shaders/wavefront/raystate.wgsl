#import /ray.wgsl

@group(2) @binding(0)
var<storage, read_write> RAY_STATES: array<RayState>;
@group(2) @binding(1)
var<storage, read_write> PATH_STATES: array<PathState>;
@group(2) @binding(2)
var<storage, read_write> SURFACE_HIT_STATES: array<SurfaceHitState>;

const LS_BSDF = 0;
const LS_LIGHT = 1;
const LS_MIS = 2;
const LS_MODE = LS_MIS;

struct RayState {
    ray: Ray,
    wavelengths: Wavelengths,
    radiance: vec4f,
    sampler_state: SamplerState,
}

struct PathState {
    throughput: vec4f,
    depth: u32,
    bsdf_pdf: f32,
    specular_bounce: u32,
    secondary_terminated: u32,
}

struct SurfaceHitState {
    bsdf: Bsdf,
    hit_pos: vec3f,
}

fn mis_weight(p1: f32, p2: f32) -> f32 {
    return p1 / (p1 + p2);
}
