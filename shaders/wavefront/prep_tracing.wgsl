#import /wavefront/queue.wgsl

@group(4) @binding(0)
var<storage, read_write> INDIRECT_SIZE: array<vec3u, 4>;

@compute
@workgroup_size(1)
fn main() {
    atomicStore(&Q_DIRECT_LIGHT.count, 0);
    atomicStore(&Q_BOUNCE.count, 0);

    let count = atomicLoad(&Q_TRACE_RAYS.count);
    INDIRECT_SIZE[0] = vec3u((count + 255) / 256, 1, 1);
}
