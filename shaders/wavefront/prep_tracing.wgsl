#import /wavefront/queue.wgsl

@group(4) @binding(0)
var<storage, read_write> INDIRECT_SIZE: array<vec3u, 3>;

@compute
@workgroup_size(1)
fn main() {
    atomicStore(&Q_DIRECT_LIGHT.count, 0);
    atomicStore(&Q_BOUNCE.count, 0);

    let count = atomicLoad(&Q_TRACE_RAYS.count);
    INDIRECT_SIZE[0] = vec3u((count + 31) / 32, 1, 1);
}
