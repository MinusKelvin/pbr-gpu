#import /scene.wgsl
#import /sampler/independent.wgsl
#import /integrator/simple.wgsl
#import /wavefront/raystate.wgsl

@compute
@workgroup_size(32)
fn main(
    @builtin(global_invocation_id) id: vec3u
) {
    if id.x >= arrayLength(&RAY_STATES) {
        return;
    }
    var state = RAY_STATES[id.x];
    SAMPLER = state.sampler_state;

    state.radiance = integrate_ray(state.wavelengths, state.ray);

    state.sampler_state = SAMPLER;
    RAY_STATES[id.x] = state;
}
