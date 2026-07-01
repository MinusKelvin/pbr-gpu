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
    if id.x >= Q_DIRECT_LIGHT.count {
        return;
    }
    let ray_id = Q_DIRECT_LIGHT.ray_ids[id.x];

    SAMPLER = RAY_STATES[ray_id].sampler_state;
    sample_direct_light(ray_id);
    RAY_STATES[ray_id].sampler_state = SAMPLER;
}

fn sample_direct_light(ray_id: u32) {
    let bsdf = SURFACE_HIT_STATES[ray_id].bsdf;
    let hit_p = SURFACE_HIT_STATES[ray_id].hit_pos;
    let outgoing = RAY_STATES[ray_id].ray.d;
    let throughput = PATH_STATES[ray_id].throughput;
    let wl = RAY_STATES[ray_id].wavelengths;

    let light_id_sample = light_sampler_sample(ROOT_LS, hit_p, sample_1d());
    if light_id_sample.pmf == 0 {
        return;
    }

    let light_sample = light_sample(light_id_sample.light, hit_p, wl, sample_2d());
    if light_sample.pdf_wrt_solid_angle == 0 {
        return;
    }

    let pdf = light_sample.pdf_wrt_solid_angle * light_id_sample.pmf;

    let bsdf_pdf = bsdf_pdf(bsdf, -outgoing, light_sample.dir)
        * f32(LS_MODE == LS_MIS);
    let contribution = throughput
        * light_sample.emission
        * bsdf_f(bsdf, -outgoing, light_sample.dir)
        * abs(dot(bsdf_normal(bsdf), light_sample.dir))
        / pdf
        * mis_weight(pdf, bsdf_pdf);

    if all(contribution == vec4f()) {
        return;
    }

    let offset = 10 * EPSILON * (1 + length(hit_p));
    let ray = Ray(hit_p + light_sample.dir * offset, light_sample.dir, RAY_STATES[ray_id].ray.time);

    let t_max = light_sample.t_max - offset - 0.0001;

    SHADOW_RAY_STATES[ray_id] = ShadowRayState(ray, contribution, t_max);
    enqueue_shadow(ray_id);
}
