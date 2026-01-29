#import /scene.wgsl
#import /ray.wgsl
#import /util/misc.wgsl
#import /material.wgsl
#import /light.wgsl

const MAX_DEPTH = 250;

fn integrate_ray(wl: Wavelengths, ray_: Ray) -> vec4f {
    var radiance = vec4f();
    var throughput = vec4f(1);

    var ray = ray_;

    var specular_bounce = false;
    var secondary_terminated = false;

    var depth = 0;
    while any(throughput > vec4f()) {
        let result = scene_raycast(ray);

        if !result.hit {
            // add infinite lights and finish
            if depth == 0 || specular_bounce {
                for (var i = 0u; i < arrayLength(&INFINITE_LIGHTS); i++) {
                    radiance += throughput * inf_light_emission(INFINITE_LIGHTS[i], ray, wl);
                }
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

        if !secondary_terminated && bsdf_terminates_secondary_wavelengths(bsdf) {
            secondary_terminated = true;
            throughput *= vec4f(4, 0, 0, 0);
        }

        // construct shading coordinate system
        let to_bsdf_frame = transpose(any_orthonormal_frame(result.n));
        let old_dir = to_bsdf_frame * -ray.d;

        // sample direct lighting
        radiance += throughput * _sample_direct_light(
            bsdf,
            to_bsdf_frame,
            old_dir,
            result,
            ray,
            wl,
        );

        // sample bsdf
        let bsdf_s = bsdf_sample(bsdf, old_dir, vec3f(sample_2d(), sample_1d()));
        if bsdf_s.pdf == 0 {
            break;
        }

        throughput *= bsdf_s.f * abs(bsdf_s.dir.z) / bsdf_s.pdf;

        // russian roulette
        let rr = max(max(throughput.x, throughput.y), max(throughput.z, throughput.w));
        if rr < 1 && depth > 1 {
            if sample_1d() >= rr {
                break;
            }
            throughput /= rr;
        }

        // spawn new ray
        let offset = 10 * EPSILON * max(1, length(result.p));
        ray.d = transpose(to_bsdf_frame) * bsdf_s.dir;
        ray.o = result.p + ray.d * offset;
        specular_bounce = bsdf_s.specular;
    }

    return radiance;
}

fn _sample_direct_light(
    bsdf: Bsdf,
    to_bsdf_frame: mat3x3f,
    old_dir: vec3f,
    hit: RaycastResult,
    ray_: Ray,
    wl: Wavelengths,
) -> vec4f {
    let light_sample = light_sampler_sample(wl, vec3f(sample_2d(), sample_1d()));
    if light_sample.pdf_wrt_solid_angle == 0 {
        return vec4f();
    }

    let new_dir = to_bsdf_frame * light_sample.dir;
    let contribution = light_sample.emission
        * bsdf_f(bsdf, old_dir, new_dir)
        * abs(new_dir.z)
        / light_sample.pdf_wrt_solid_angle;

    if all(contribution == vec4f()) {
        return vec4f();
    }

    var ray = ray_;
    let offset = copysign(10, new_dir.z) * EPSILON * max(1, length(hit.p));
    ray.d = light_sample.dir;
    ray.o = hit.p + ray.d * offset;

    if scene_raycast(ray).hit {
        return vec4f();
    }

    return contribution;
}
