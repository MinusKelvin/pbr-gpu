#import /scene.wgsl
#import /sampler/independent.wgsl
#import /integrator/simple.wgsl
#import /wavefront/raystate.wgsl
#import /film.wgsl

@compute
@workgroup_size(32)
fn main(
    @builtin(global_invocation_id) id: vec3u
) {
    var state = RAY_STATES[id.x];
    if state.exists == 0 {
        return;
    }

    film_add_sample(
        state.px,
        state.wavelengths,
        state.radiance / film_wavelengths_pdf(state.wavelengths)
    );
}
