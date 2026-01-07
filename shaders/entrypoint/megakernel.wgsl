#import /util.wgsl
#import /scene.wgsl
#import /sampler.wgsl
#import /camera.wgsl
#import /film.wgsl
#import /filter.wgsl

struct Immediates {
    sample_number: u32
}

var<immediate> imm: Immediates;

@compute
@workgroup_size(1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    sample_init(id.xy, imm.sample_number);

    let wavelengths = film_wavelengths_sample();
    let fs = filter_sample();
    var film_position_norm = (vec2f(id.xy) + fs.p + 0.5) / vec2f(film_size());
    film_position_norm.y = 1 - film_position_norm.y;
    let film_position_ndc = 2 * film_position_norm - 1;

    let ray = camera_sample_ray(film_position_ndc);

    let result = shape_raycast(ShapeId(0), ray, FLOAT_MAX);

    let d = dot(result.n, ray.d);
    let value = select(
        vec3(d, d / 2, 0),
        vec3(0, -d / 2, -d),
        d < 0
    );

    // film_add_sample(id.xy, wavelengths, vec4(abs(ray.d), 1.0));
    film_add_sample(id.xy, wavelengths, vec4(value, 1.0));
}
