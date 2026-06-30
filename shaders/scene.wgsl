#import /shapes.wgsl
#import /transform.wgsl

@group(0) @binding(3)
var<storage> TRI_PROPERTIES: array<TriProperty>;
@group(0) @binding(4)
var<storage> TRI_INDEX_OFFSETS: array<u32>;

@group(0) @binding(32)
var SCENE_AS: acceleration_structure;

struct TriProperty {
    material: MaterialId,
    light: LightId,
    alpha: FloatTextureId,
}

fn scene_raycast(ray: Ray, max_t: f32) -> RaycastResult {
    let ray_desc = RayDesc(0, ~0u, 0, max_t, ray.o, ray.d);
    var tracer: ray_query;

    rayQueryInitialize(&tracer, SCENE_AS, ray_desc);

    while rayQueryProceed(&tracer) {
        let info = rayQueryGetCandidateIntersection(&tracer);
        rayQueryConfirmIntersection(&tracer);
    }

    let hit = rayQueryGetCommittedIntersection(&tracer);
    if hit.kind == RAY_QUERY_INTERSECTION_NONE {
        return RaycastResult();
    }

    let tri_id = TRI_INDEX_OFFSETS[hit.instance_custom_data + hit.geometry_index]
        + hit.primitive_index;
    let b = hit.barycentrics;
    let bary = vec3f(1 - b.x - b.y, b);

    var result = triangle_raycast_result(TRIANGLES[tri_id], bary, hit.t);
    result.material = TRI_PROPERTIES[tri_id].material;
    result.light = TRI_PROPERTIES[tri_id].light;
    result.p = hit.object_to_world * vec4(result.p, 1);
    result.n = (transpose(hit.world_to_object) * result.n).xyz;
    result.tangent = hit.object_to_world * vec4(result.tangent, 0);

    return result;
}
