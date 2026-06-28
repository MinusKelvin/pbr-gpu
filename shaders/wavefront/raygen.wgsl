#import /sampler/independent.wgsl
#import /camera.wgsl
#import /film.wgsl
#import /filter.wgsl
#import /wavefront/raystate.wgsl
#import /light_sampler.wgsl
#import /scene.wgsl

struct Immediates {
    sample_number: u32
}

var<immediate> imm: Immediates;

@compute
@workgroup_size(8, 4)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let size = film_size();
    if any(id.xy >= film_size()) {
        return;
    }
    let ray_id = id.x + id.y * size.x;

    sample_init(id.xy, imm.sample_number);

    let wavelengths = film_wavelengths_sample();
    let fs = filter_sample();
    var film_position_norm = (vec2f(id.xy) + fs.p + 0.5) / vec2f(film_size());
    film_position_norm.y = 1 - film_position_norm.y;
    let film_position_ndc = 2 * film_position_norm - 1;

    let ray = camera_sample_ray(film_position_ndc);

    RAY_STATES[ray_id] = RayState(
        ray,
        wavelengths,
        vec4f(),
        id.xy,
        SAMPLER,
    );

    PATH_STATES[ray_id] = PathState(vec4f(1), 0, 0, 0, 0);
}
