#import /ray.wgsl
#import shapes/sphere.wgsl
#import shapes/triangle.wgsl

@group(0) @binding(0)
var<storage> SPHERES: array<Sphere>;
@group(0) @binding(1)
var<storage> TRIANGLES: array<Triangle>;

const SHAPE_TAG_BITS: u32 = 1;
const SHAPE_TAG_SHIFT: u32 = 32 - SHAPE_TAG_BITS;
const SHAPE_IDX_MASK: u32 = (1 << SHAPE_TAG_SHIFT) - 1;
const SHAPE_TAG_MASK: u32 = ~SHAPE_IDX_MASK;

const SHAPE_SPHERE: u32 = 0 << SHAPE_TAG_SHIFT;
const SHAPE_TRIANGLE: u32 = 1 << SHAPE_TAG_SHIFT;

struct ShapeId {
    id: u32
}

fn shape_raycast(shape: ShapeId, ray: Ray, t_max: f32) -> RaycastResult {
    switch shape.id & SHAPE_TAG_MASK {
        case SHAPE_SPHERE {
            return sphere_raycast(SPHERES[shape.id & SHAPE_IDX_MASK], ray, t_max);
        }
        case SHAPE_TRIANGLE {
            return triangle_raycast(TRIANGLES[shape.id & SHAPE_IDX_MASK], ray, t_max);
        }
        default {
            // unreachable
            return RaycastResult();
        }
    }
}

struct ShapeSample {
    p: vec3f,
    ng: vec3f,
    pdf_wrt_area: f32,
}

fn shape_sample(shape: ShapeId, p: vec3f, random: vec2f) -> ShapeSample {
    switch shape.id & SHAPE_TAG_MASK {
        case SHAPE_SPHERE {
            return sphere_sample(SPHERES[shape.id & SHAPE_IDX_MASK], p, random);
        }
        case SHAPE_TRIANGLE {
            return triangle_sample(TRIANGLES[shape.id & SHAPE_IDX_MASK], p, random);
        }
        default {
            // unreachable
            return ShapeSample();
        }
    }
}
