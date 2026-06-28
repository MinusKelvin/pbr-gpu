#import /wavefront/queue.wgsl

@compute
@workgroup_size(1)
fn main() {
    atomicStore(&Q_TRACE_RAYS.count, 0);
}
