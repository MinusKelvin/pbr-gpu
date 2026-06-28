#import /scene.wgsl
#import /sampler/independent.wgsl
#import /wavefront/raystate.wgsl
#import /film.wgsl
#import /light_sampler.wgsl

@compute
@workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) id: vec3u
) {
    let size = film_size();
    if any(id.xy >= film_size()) {
        return;
    }
    let ray_id = id.x + id.y * size.x;

    let wl = RAY_STATES[ray_id].wavelengths;

    film_add_sample(id.xy, wl, RAY_STATES[ray_id].radiance / film_wavelengths_pdf(wl));
}
