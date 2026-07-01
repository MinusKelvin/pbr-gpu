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
    if id.x >= Q_SHADOW.count {
        return;
    }
    let ray_id = Q_SHADOW.ray_ids[id.x];

    shadow_ray(ray_id);
}

fn shadow_ray(ray_id: u32) {
    let ray = SHADOW_RAY_STATES[ray_id].ray;
    let t_max = SHADOW_RAY_STATES[ray_id].t_max;
    let contribution = SHADOW_RAY_STATES[ray_id].contribution;

    if scene_raycast(ray, t_max).hit {
        return;
    }

    RAY_STATES[ray_id].radiance += contribution;
}
