#import /scene.wgsl
#import /sampler/independent.wgsl
#import /wavefront/raystate.wgsl
#import /film.wgsl
#import /light_sampler.wgsl

@compute
@workgroup_size(32)
fn main(
    @builtin(global_invocation_id) id: vec3u
) {
    if id.x >= arrayLength(&RAY_STATES) {
        return;
    }
    let state = RAY_STATES[id.x];

    film_add_sample(
        state.px,
        state.wavelengths,
        state.radiance / film_wavelengths_pdf(state.wavelengths)
    );
}
