#import /scene.wgsl
#import /ray.wgsl
#import /util.wgsl
#import /material.wgsl

const MAX_DEPTH = 250;

fn integrate_ray(wl: Wavelengths, ray_: Ray) -> vec4f {
    var radiance = vec4f();
    var throughput = vec4f(1);

    var ray = ray_;

    var depth = 0;
    while any(throughput > vec4f()) {
        let result = scene_raycast(ray);

        if !result.hit {
            radiance += throughput * spectrum_sample(SPECTRUM_D65_1NIT, wl);
            break;
        }

        // enforce termination
        depth += 1;
        if depth > MAX_DEPTH {
            break;
        }

        let bsdf = material_evaluate(result.material, result.uv, wl);

        // construct shading coordinate system
        let to_bsdf_frame = transpose(any_orthonormal_frame(result.n));

        let old_dir = to_bsdf_frame * -ray.d;
        let new_dir = sample_uniform_sphere(sample_2d());

        // evaluate bsdf
        throughput *= bsdf_f(bsdf, old_dir, new_dir)
            * abs(new_dir.z)
            / (1 / (2 * TWO_PI));

        // spawn new ray
        let offset = copysign(10, new_dir.z) * EPSILON * max(1, length(result.p));
        ray.o = result.p + result.n * offset;
        ray.d = transpose(to_bsdf_frame) * new_dir;
    }

    return radiance;
}
