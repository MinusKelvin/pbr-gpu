#import scene/shapes.wgsl

@group(0) @binding(32)
var<storage> BVH_ROOT: u32;
@group(0) @binding(33)
var<storage> BVH_NODES: array<BvhNode>;
@group(0) @binding(34)
var<storage> TRANSFORM_NODES: array<TransformNode>;
@group(0) @binding(35)
var<storage> PRIMITIVE_NODES: array<PrimitiveNode>;

const NODE_TAG_BITS: u32 = 2;
const NODE_TAG_SHIFT: u32 = 32 - NODE_TAG_BITS;
const NODE_IDX_MASK: u32 = (1 << NODE_TAG_SHIFT) - 1;
const NODE_TAG_MASK: u32 = ~NODE_IDX_MASK;
const NODE_TAG_BVH: u32 = 0 << NODE_TAG_SHIFT;
const NODE_TAG_TRANSFORM: u32 = 1 << NODE_TAG_SHIFT;
const NODE_TAG_PRIMITIVE: u32 = 2 << NODE_TAG_SHIFT;

struct NodeId {
    id: u32,
}

struct BvhNode {
    min: vec3f,
    flags: u32,
    max: vec3f,
    far_node: NodeId,
}

const BVH_FLAG_X: u32 = 1 << 0;
const BVH_FLAG_Y: u32 = 1 << 1;
const BVH_FLAG_Z: u32 = 1 << 2;

struct TransformNode {
    transform: Transform,
    object: NodeId,
}

struct PrimitiveNode {
    shape: ShapeId,
}

struct TransformStackEntry {
    old_ray: Ray,
}

fn scene_raycast(ray_: Ray) -> RaycastResult {
    var closest: RaycastResult;
    closest.t = FLOAT_MAX;

    var ray = ray_;
    var inv_ray_dir = 1 / ray.d;

    var bvh_stack: array<NodeId, 64>;
    var i = 0;
    bvh_stack[0] = NodeId(BVH_ROOT);

    var transform_stack: array<TransformStackEntry, 2>;
    var transform_i = 0;

    const POP_TRANSFORM_SENTINEL: u32 = ~0u;

    while i >= 0 {
        if bvh_stack[i].id == POP_TRANSFORM_SENTINEL {
            ray = transform_stack[transform_i].old_ray;
            inv_ray_dir = 1 / ray.d;
            transform_i -= 1;
            i -= 1;
            continue;
        }

        switch (bvh_stack[i].id & NODE_TAG_MASK) {
            case NODE_TAG_BVH {
                let node = BVH_NODES[bvh_stack[i].id];
                let t0 = (node.min - ray.o) * inv_ray_dir;
                let t1 = (node.max - ray.o) * inv_ray_dir;
                let t_near = min(t0, t1);
                let t_far = max(t0, t1);
                let t_enter = max(max(t_near.x, t_near.y), t_near.z);
                let t_exit = min(min(t_far.x, t_far.y), t_far.z);

                if t_enter >= closest.t || t_enter > t_exit {
                    i -= 1;
                } else if (node.far_node.id & NODE_TAG_MASK) == NODE_TAG_BVH {
                    bvh_stack[i] = NodeId(bvh_stack[i].id + 1);
                    bvh_stack[i + 1] = node.far_node;
                    i += 1;
                } else {
                    bvh_stack[i] = node.far_node;
                }
            }
            case NODE_TAG_TRANSFORM {
                // todo
                i -= 1;
            }
            case NODE_TAG_PRIMITIVE {
                let shape = PRIMITIVE_NODES[bvh_stack[i].id & NODE_IDX_MASK].shape;
                let result = shape_raycast(shape, ray, closest.t);
                if result.hit {
                    closest = result;
                }
                i -= 1;
            }
            default {
                // unreachable
                return RaycastResult();
            }
        }
    }

    return closest;
}
