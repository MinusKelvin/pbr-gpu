#import /wavefront/queue.wgsl

@compute
@workgroup_size(1)
fn main() {
    atomicStore(&ACTIVE_RAYS_NEXT.count, 0);
}
