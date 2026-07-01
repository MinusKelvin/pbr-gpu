#import /scene.wgsl
#import /sampler/independent.wgsl
#import /wavefront/raystate.wgsl
#import /wavefront/queue.wgsl
#import /ray.wgsl
#import /util/misc.wgsl
#import /material.wgsl
#import /light.wgsl
#import /light_sampler.wgsl

@compute
@workgroup_size(256)
fn main(
    @builtin(global_invocation_id) id: vec3u
) {
    if id.x >= Q_TRACE_RAYS.count {
        return;
    }
    let ray_id = Q_TRACE_RAYS.ray_ids[id.x];

    SAMPLER = RAY_STATES[ray_id].sampler_state;

    trace_ray(ray_id);

    RAY_STATES[ray_id].sampler_state = SAMPLER;
}

fn trace_ray(ray_id: u32) {
    let ray = RAY_STATES[ray_id].ray;
    let wl = RAY_STATES[ray_id].wavelengths;

    var throughput = PATH_STATES[ray_id].throughput;
    let specular_bounce = PATH_STATES[ray_id].specular_bounce != 0;
    var secondary_terminated = PATH_STATES[ray_id].secondary_terminated != 0;
    let bsdf_pdf = PATH_STATES[ray_id].bsdf_pdf;
    var depth = PATH_STATES[ray_id].depth;

    let result = scene_raycast(ray, FLOAT_MAX);

    if !result.hit {
        // add infinite lights and finish
        if depth == 0 || specular_bounce || LS_MODE == LS_BSDF {
            for (var i = 1u; i < arrayLength(&INFINITE_LIGHTS); i++) {
                RAY_STATES[ray_id].radiance += throughput * inf_light_emission(INFINITE_LIGHTS[i], ray, wl);
            }
        }
        if depth > 0 && !specular_bounce && LS_MODE == LS_MIS {
            // direct lighting MIS
            for (var i = 1u; i < arrayLength(&INFINITE_LIGHTS); i++) {
                let ls_pdf = light_sampler_pmf(ROOT_LS, ray.o, INFINITE_LIGHTS[i])
                    * light_pdf(INFINITE_LIGHTS[i], ray.o, ray.d);
                RAY_STATES[ray_id].radiance += throughput
                    * inf_light_emission(INFINITE_LIGHTS[i], ray, wl)
                    * mis_weight(bsdf_pdf, ls_pdf);
            }
        }
        return;
    }

    // add light emitted by surface
    if depth == 0 || specular_bounce || LS_MODE == LS_BSDF {
        RAY_STATES[ray_id].radiance += throughput * light_emission(result.light, ray, result, wl);
    }
    if depth > 0 && !specular_bounce && LS_MODE == LS_MIS {
        // direct lighting MIS
        let ls_pdf = light_sampler_pmf(ROOT_LS, ray.o, result.light)
            * light_pdf(result.light, ray.o, ray.d);
        RAY_STATES[ray_id].radiance += throughput
            * light_emission(result.light, ray, result, wl)
            * mis_weight(bsdf_pdf, ls_pdf);
    }

    let bsdf = material_evaluate(result.material, result, wl);

    if !secondary_terminated && bsdf_terminates_secondary_wavelengths(bsdf) {
        secondary_terminated = true;
        throughput *= vec4f(4, 0, 0, 0);
    }

    PATH_STATES[ray_id].throughput = throughput;
    PATH_STATES[ray_id].depth = depth + 1;
    PATH_STATES[ray_id].secondary_terminated = u32(secondary_terminated);
    SURFACE_HIT_STATES[ray_id] = SurfaceHitState(bsdf, result.p);

    enqueue_bounce(ray_id);
    if LS_MODE != LS_BSDF {
        enqueue_direct_light(ray_id);
    } else {
        // consume sampler dimensions which are used by direct light sampling
        sample_2d();
        sample_1d();
    }
}
