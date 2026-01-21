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
@workgroup_size(8, 4)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    if any(id.xy >= film_size()) {
        return;
    }

    sample_init(id.xy, imm.sample_number);

    let wavelengths = film_wavelengths_sample();
    let fs = filter_sample();
    var film_position_norm = (vec2f(id.xy) + fs.p + 0.5) / vec2f(film_size());
    film_position_norm.y = 1 - film_position_norm.y;
    let film_position_ndc = 2 * film_position_norm - 1;

    let ray = camera_sample_ray(film_position_ndc);

    let result = scene_raycast(ray);

    let d = f32(result.hit) * max(0.1, dot(result.n, -ray.d));

    let value = d * spectrum_sample(SPECTRUM_D65_1NIT, wavelengths)
        / film_wavelengths_pdf(wavelengths);
    film_add_sample(id.xy, wavelengths, value);
}
