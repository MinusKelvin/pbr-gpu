#import /scene.wgsl
#import /sampler/independent.wgsl
#import /wavefront/raystate.wgsl
#import /ray.wgsl
#import /util/misc.wgsl
#import /material.wgsl
#import /light.wgsl
#import /light_sampler.wgsl

@compute
@workgroup_size(32)
fn main(
    @builtin(global_invocation_id) id: vec3u
) {
    if id.x >= arrayLength(&RAY_STATES) {
        return;
    }
    SAMPLER = RAY_STATES[id.x].sampler_state;

    integrate_ray(id.x);

    RAY_STATES[id.x].sampler_state = SAMPLER;
}

const MAX_DEPTH = 31;
const LS_BSDF = 0;
const LS_LIGHT = 1;
const LS_MIS = 2;
const LS_MODE = LS_MIS;

fn integrate_ray(ray_id: u32) {
    var throughput = PATH_STATES[ray_id].throughput;
    if all(throughput == vec4f()) {
        return;
    }

    var ray = RAY_STATES[ray_id].ray;
    let wl = RAY_STATES[ray_id].wavelengths;

    var specular_bounce = PATH_STATES[ray_id].specular_bounce != 0;
    var secondary_terminated = PATH_STATES[ray_id].secondary_terminated != 0;
    var bsdf_pdf = PATH_STATES[ray_id].bsdf_pdf;
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
        PATH_STATES[ray_id].throughput = vec4f();
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

    // enforce termination
    depth += 1;
    if depth > MAX_DEPTH {
        PATH_STATES[ray_id].throughput = vec4f();
        return;
    }

    let bsdf = material_evaluate(result.material, result, wl);

    if !secondary_terminated && bsdf_terminates_secondary_wavelengths(bsdf) {
        secondary_terminated = true;
        throughput *= vec4f(4, 0, 0, 0);
    }

    if LS_MODE != LS_BSDF {
        // sample direct lighting
        RAY_STATES[ray_id].radiance += throughput * _sample_direct_light(
            bsdf,
            result,
            ray,
            wl,
        );
    } else {
        // consume sampler dimensions which would be used by direct light sampling
        sample_2d();
        sample_1d();
    }

    // sample bsdf
    let bsdf_s = bsdf_sample(bsdf, -ray.d, vec3f(sample_2d(), sample_1d()));
    if bsdf_s.pdf == 0 {
        PATH_STATES[ray_id].throughput = vec4f();
        return;
    }

    bsdf_pdf = bsdf_s.pdf;

    throughput *= bsdf_s.f * abs(dot(bsdf_normal(bsdf), bsdf_s.dir)) / bsdf_s.pdf;

    // russian roulette
    let rr = max(max(throughput.x, throughput.y), max(throughput.z, throughput.w));
    if rr < 1 && depth > 1 {
        if sample_1d() > rr {
            PATH_STATES[ray_id].throughput = vec4f();
            return;
        }
        throughput /= rr;
    }

    // spawn new ray
    let offset = 10 * EPSILON * (1 + length(result.p));
    ray.d = bsdf_s.dir;
    ray.o = result.p + ray.d * offset;
    specular_bounce = bsdf_s.specular;

    RAY_STATES[ray_id].ray = ray;
    PATH_STATES[ray_id] = PathState(
        throughput, depth, bsdf_pdf, u32(specular_bounce), u32(secondary_terminated)
    );
}

fn _sample_direct_light(
    bsdf: Bsdf,
    hit: RaycastResult,
    ray_: Ray,
    wl: Wavelengths,
) -> vec4f {
    let light_id_sample = light_sampler_sample(ROOT_LS, hit.p, sample_1d());
    if light_id_sample.pmf == 0 {
        return vec4f();
    }

    let light_sample = light_sample(light_id_sample.light, hit.p, wl, sample_2d());
    if light_sample.pdf_wrt_solid_angle == 0 {
        return vec4f();
    }

    let pdf = light_sample.pdf_wrt_solid_angle * light_id_sample.pmf;

    let bsdf_pdf = bsdf_pdf(bsdf, -ray_.d, light_sample.dir)
        * f32(LS_MODE == LS_MIS);
    let contribution = light_sample.emission
        * bsdf_f(bsdf, -ray_.d, light_sample.dir)
        * abs(dot(bsdf_normal(bsdf), light_sample.dir))
        / pdf
        * mis_weight(pdf, bsdf_pdf);

    if all(contribution == vec4f()) {
        return vec4f();
    }

    var ray = ray_;
    let offset = 10 * EPSILON * (1 + length(hit.p));
    ray.d = light_sample.dir;
    ray.o = hit.p + ray.d * offset;

    if scene_raycast(ray, light_sample.t_max - offset - 0.0001).hit {
        return vec4f();
    }

    return contribution;
}

fn mis_weight(p1: f32, p2: f32) -> f32 {
    return p1 / (p1 + p2);
}
