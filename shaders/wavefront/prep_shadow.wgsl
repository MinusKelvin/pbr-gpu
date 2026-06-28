#import /wavefront/queue.wgsl

@group(4) @binding(0)
var<storage, read_write> INDIRECT_SIZE: array<vec3u, 4>;

@compute
@workgroup_size(1)
fn main() {
    let num_shadow = atomicLoad(&Q_SHADOW.count);
    INDIRECT_SIZE[3] = vec3u((num_shadow + 31) / 32, 1, 1);
}
