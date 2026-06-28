@group(3) @binding(0)
var<storage, read_write> Q_TRACE_RAYS: Queue;
@group(3) @binding(1)
var<storage, read_write> Q_DIRECT_LIGHT: Queue;
@group(3) @binding(2)
var<storage, read_write> Q_BOUNCE: Queue;
@group(3) @binding(3)
var<storage, read_write> Q_SHADOW: Queue;

struct Queue {
    count: atomic<u32>,
    ray_ids: array<u32>,
}

fn enqueue_trace(ray_id: u32) {
    let idx = atomicAdd(&Q_TRACE_RAYS.count, 1);
    Q_TRACE_RAYS.ray_ids[idx] = ray_id;
}

fn enqueue_direct_light(ray_id: u32) {
    let idx = atomicAdd(&Q_DIRECT_LIGHT.count, 1);
    Q_DIRECT_LIGHT.ray_ids[idx] = ray_id;
}

fn enqueue_bounce(ray_id: u32) {
    let idx = atomicAdd(&Q_BOUNCE.count, 1);
    Q_BOUNCE.ray_ids[idx] = ray_id;
}

fn enqueue_shadow(ray_id: u32) {
    let idx = atomicAdd(&Q_SHADOW.count, 1);
    Q_SHADOW.ray_ids[idx] = ray_id;
}
