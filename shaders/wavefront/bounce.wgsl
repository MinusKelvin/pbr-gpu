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
    if id.x >= Q_BOUNCE.count {
        return;
    }
    let ray_id = Q_BOUNCE.ray_ids[id.x];

    SAMPLER = RAY_STATES[ray_id].sampler_state;
    bounce_ray(ray_id);
    RAY_STATES[ray_id].sampler_state = SAMPLER;
}

fn bounce_ray(ray_id: u32) {
    var ray = RAY_STATES[ray_id].ray;
    let wl = RAY_STATES[ray_id].wavelengths;

    var throughput = PATH_STATES[ray_id].throughput;
    let depth = PATH_STATES[ray_id].depth;

    let bsdf = SURFACE_HIT_STATES[ray_id].bsdf;
    let hit_pos = SURFACE_HIT_STATES[ray_id].hit_pos;

    // sample bsdf
    let bsdf_s = bsdf_sample(bsdf, -ray.d, vec3f(sample_2d(), sample_1d()));
    if bsdf_s.pdf == 0 {
        return;
    }

    throughput *= bsdf_s.f * abs(dot(bsdf_normal(bsdf), bsdf_s.dir)) / bsdf_s.pdf;

    // russian roulette
    let rr = max(max(throughput.x, throughput.y), max(throughput.z, throughput.w));
    if rr < 1 && depth > 1 {
        if sample_1d() > rr {
            return;
        }
        throughput /= rr;
    }

    // spawn new ray
    let offset = 10 * EPSILON * (1 + length(hit_pos));
    ray.d = bsdf_s.dir;
    ray.o = hit_pos + ray.d * offset;

    RAY_STATES[ray_id].ray = ray;
    PATH_STATES[ray_id].throughput = throughput;
    PATH_STATES[ray_id].bsdf_pdf = bsdf_s.pdf;
    PATH_STATES[ray_id].specular_bounce = u32(bsdf_s.specular);

    enqueue_trace(ray_id);
}
