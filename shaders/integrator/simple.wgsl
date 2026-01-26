#import /scene.wgsl
#import /ray.wgsl
#import /util/misc.wgsl
#import /material.wgsl
#import /light.wgsl

const MAX_DEPTH = 25;

fn integrate_ray(wl: Wavelengths, ray_: Ray) -> vec4f {
    var radiance = vec4f();
    var throughput = vec4f(1);

    var ray = ray_;

    var depth = 0;
    while any(throughput > vec4f()) {
        let result = scene_raycast(ray);

        if !result.hit {
            // add infinite lights and finish
            for (var i = 0u; i < arrayLength(&INFINITE_LIGHTS); i++) {
                radiance += throughput * inf_light_emission(INFINITE_LIGHTS[i], ray, wl);
            }
            break;
        }

        // add light emitted by surface
        radiance += throughput * light_emission(result.light, result, wl);

        // enforce termination
        depth += 1;
        if depth > MAX_DEPTH {
            break;
        }

        let bsdf = material_evaluate(result.material, result.uv, wl);

        // construct shading coordinate system
        let to_bsdf_frame = transpose(any_orthonormal_frame(result.n));

        let old_dir = to_bsdf_frame * -ray.d;

        // sample bsdf
        let bsdf_s = bsdf_sample(bsdf, old_dir, vec3f(sample_2d(), 0.0));
        if bsdf_s.pdf == 0 {
            break;
        }

        throughput *= bsdf_s.f * abs(bsdf_s.dir.z) / bsdf_s.pdf;

        // spawn new ray
        let offset = copysign(10, bsdf_s.dir.z) * EPSILON * max(1, length(result.p));
        ray.o = result.p + result.n * offset;
        ray.d = transpose(to_bsdf_frame) * bsdf_s.dir;
    }

    return radiance;
}
