#import /scene.wgsl
#import /ray.wgsl
#import /util/distr.wgsl
#import /material.wgsl
#import /light.wgsl

const MAX_DEPTH = 25;

fn integrate_ray(wl: Wavelengths, ray_: Ray) -> vec4f {
    var radiance = vec4f();
    var throughput = vec4f(1);

    var ray = ray_;

    var depth = 0;
    while any(throughput > vec4f()) {
        let result = scene_raycast(ray, FLOAT_MAX);

        if !result.hit {
            // add infinite lights and finish
            for (var i = 1u; i < arrayLength(&INFINITE_LIGHTS); i++) {
                radiance += throughput * inf_light_emission(INFINITE_LIGHTS[i], ray, wl);
            }
            break;
        }

        // add light emitted by surface
        radiance += throughput * light_emission(result.light, ray, result, wl);

        // enforce termination
        depth += 1;
        if depth > MAX_DEPTH {
            break;
        }

        let bsdf = material_evaluate(result.material, result, wl);

        let new_dir = sample_uniform_sphere(sample_2d());

        // evaluate bsdf
        throughput *= bsdf_f(bsdf, -ray.d, new_dir)
            * abs(dot(bsdf_normal(bsdf), new_dir))
            / (1 / (2 * TWO_PI));

        // spawn new ray
        let offset = 10 * EPSILON * (1 + length(result.p));
        ray.d = new_dir;
        ray.o = result.p + ray.d * offset;
    }

    return radiance;
}
