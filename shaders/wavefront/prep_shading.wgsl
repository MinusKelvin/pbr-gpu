#import /wavefront/queue.wgsl

@group(4) @binding(0)
var<storage, read_write> INDIRECT_SIZE: array<vec3u, 4>;

@compute
@workgroup_size(1)
fn main() {
    atomicStore(&Q_TRACE_RAYS.count, 0);
    atomicStore(&Q_SHADOW.count, 0);

    let num_direct_light = atomicLoad(&Q_DIRECT_LIGHT.count);
    INDIRECT_SIZE[1] = vec3u((num_direct_light + 31) / 32, 1, 1);

    let num_bounce = atomicLoad(&Q_BOUNCE.count);
    INDIRECT_SIZE[2] = vec3u((num_bounce + 31) / 32, 1, 1);
}
