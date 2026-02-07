#import /scene.wgsl
#import /sampler/meta.wgsl
#import /camera.wgsl
#import /film.wgsl
#import /filter.wgsl
#import /integrator/meta.wgsl

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

    let radiance = integrate_ray(wavelengths, ray);

    film_add_sample(id.xy, wavelengths, radiance / film_wavelengths_pdf(wavelengths));
}
