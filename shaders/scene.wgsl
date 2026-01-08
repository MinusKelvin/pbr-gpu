#import scene/shapes.wgsl

fn scene_raycast(ray: Ray) -> RaycastResult {
    var closest: RaycastResult;
    closest.t = FLOAT_MAX;
    for (var i = 0u; i < arrayLength(&SPHERES); i++) {
        let result = shape_raycast(ShapeId(TAG_SPHERE | i), ray, closest.t);
        if result.hit {
            closest = result;
        }
    }
    for (var i = 0u; i < arrayLength(&TRIANGLES); i++) {
        let result = shape_raycast(ShapeId(TAG_TRIANGLE | i), ray, closest.t);
        if result.hit {
            closest = result;
        }
    }
    return closest;
}
