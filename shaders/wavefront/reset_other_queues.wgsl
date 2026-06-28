#import /wavefront/queue.wgsl

@compute
@workgroup_size(1)
fn main() {
    atomicStore(&Q_DIRECT_LIGHT.count, 0);
    atomicStore(&Q_BOUNCE.count, 0);
}
