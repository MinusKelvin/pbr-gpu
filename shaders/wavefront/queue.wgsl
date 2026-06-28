@group(3) @binding(0)
var<storage, read> ACTIVE_RAYS: QueueRead;
@group(3) @binding(1)
var<storage, read_write> ACTIVE_RAYS_NEXT: Queue;

struct Queue {
    count: atomic<u32>,
    ray_ids: array<u32>,
}

struct QueueRead {
    count: u32,
    ray_ids: array<u32>,
}

fn enqueue_ray(ray_id: u32) {
    let idx = atomicAdd(&ACTIVE_RAYS_NEXT.count, 1);
    ACTIVE_RAYS_NEXT.ray_ids[idx] = ray_id;
}
