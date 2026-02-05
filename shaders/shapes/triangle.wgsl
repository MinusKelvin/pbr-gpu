#import /ray.wgsl

@group(0) @binding(2)
var<storage> TRI_VERTICES: array<TriVertex>;

struct Triangle {
    v0: u32,
    v1: u32,
    v2: u32,
}

struct TriVertex {
    p: vec3f,
    u: f32,
    n: vec3f,
    v: f32,
}

struct TriHit {
    hit: bool,
    b: vec3f,
    t: f32,
}

fn triangle_hit(v0: vec3f, v1: vec3f, v2: vec3f, ray: Ray, t_max: f32) -> TriHit {
    let to_ray_triangle_space = triangle_ray_transform(ray);

    let p0 = (to_ray_triangle_space * vec4(v0, 1)).xyz;
    let p1 = (to_ray_triangle_space * vec4(v1, 1)).xyz;
    let p2 = (to_ray_triangle_space * vec4(v2, 1)).xyz;

    // e_n is computed using the edge that vertex n does not participate in, so that it is zero when
    // the barycentric coordinate of the origin is zero.
    let e = vec3(
        edge_function(p1, p2),
        edge_function(p2, p0),
        edge_function(p0, p1),
    );
    let det = dot(e, vec3f(1));

    if any(e < vec3f()) && any(e > vec3f()) || det == 0 {
        return TriHit();
    }

    // barycentric coordinates of triangle hit
    let b = e / det;

    // z is ray t in the ray-triangle intersection space
    let t = dot(b, vec3(p0.z, p1.z, p2.z));

    if t <= 0 || t > t_max {
        return TriHit();
    }

    return TriHit(true, b, t);
}

fn triangle_raycast(tri: Triangle, ray: Ray, t_max: f32) -> RaycastResult {
    let v0 = TRI_VERTICES[tri.v0];
    let v1 = TRI_VERTICES[tri.v1];
    let v2 = TRI_VERTICES[tri.v2];

    let hit = triangle_hit(v0.p, v1.p, v2.p, ray, t_max);
    if !hit.hit {
        return RaycastResult();
    }

    let p = hit.b.x * v0.p
        + hit.b.y * v1.p
        + hit.b.z * v2.p;

    var n_shade = hit.b.x * v0.n
        + hit.b.y * v1.n
        + hit.b.z * v2.n;

    let uv = hit.b.x * vec2(v0.u, v0.v)
        + hit.b.y * vec2(v1.u, v1.v)
        + hit.b.z * vec2(v2.u, v2.v);

    var n_geo = normalize(cross(v1.p - v0.p, v2.p - v0.p));

    if all(n_shade == vec3f()) {
        n_shade = n_geo;
    } else {
        n_shade = normalize(n_shade);
        if dot(n_geo, n_shade) < 0 {
            n_geo = -n_geo;
        }
    }

    return RaycastResult(true, p, n_shade, n_geo, hit.t, MaterialId(), LightId(), uv);
}

fn edge_function(p0: vec3f, p1: vec3f) -> f32 {
    return difference_of_products(p0.x, p1.y, p1.x, p0.y);
}

fn triangle_ray_transform(ray: Ray) -> mat4x4f {
    // computes matrix for transforming into the ray-triangle intersection space,
    // in which the ray starts at the origin and goes in the Z+ direction
    let translate = mat4x4(
        vec4f(1, 0, 0, 0),
        vec4f(0, 1, 0, 0),
        vec4f(0, 0, 1, 0),
        vec4f(-ray.o, 1),
    );

    var permute: mat4x4f;
    let absd = abs(ray.d);
    if all(absd.xx >= absd.yz) {
        permute = mat4x4(
            vec4f(0, 0, 1, 0),
            vec4f(1, 0, 0, 0),
            vec4f(0, 1, 0, 0),
            vec4f(0, 0, 0, 1),
        );
    } else if all(absd.yy > absd.xz) {
        permute = mat4x4(
            vec4f(1, 0, 0, 0),
            vec4f(0, 0, 1, 0),
            vec4f(0, 1, 0, 0),
            vec4f(0, 0, 0, 1),
        );
    } else {
        permute = mat4x4(
            vec4f(1, 0, 0, 0),
            vec4f(0, 1, 0, 0),
            vec4f(0, 0, 1, 0),
            vec4f(0, 0, 0, 1),
        );
    }

    let d = (permute * vec4f(ray.d, 0)).xyz;
    let shear = mat4x4(
        vec4f(1, 0, 0, 0),
        vec4f(0, 1, 0, 0),
        vec4f(-d.xy, 1, 0) / d.z,
        vec4f(0, 0, 0, 1),
    );

    return shear * permute * translate;
}

fn triangle_sample(tri: Triangle, ref_p: vec3f, random: vec2f) -> ShapeSample {
    let v0 = TRI_VERTICES[tri.v0];
    let v1 = TRI_VERTICES[tri.v1];
    let v2 = TRI_VERTICES[tri.v2];

    // better distribution than mirroring across y=1-x wrt low-discrepancy sampling (via pbr-book)
    var b: vec3f;
    if random.x < random.y {
        b.x = random.x / 2;
        b.y = random.y - b.x;
    } else {
        b.y = random.y / 2;
        b.x = random.x - b.y;
    }
    b.z = 1 - b.x - b.y;

    let p = b.x * v0.p
        + b.y * v1.p
        + b.z * v2.p;

    var n_shade = b.x * v0.n
        + b.y * v1.n
        + b.z * v2.n;

    let uv = b.x * vec2(v0.u, v0.v)
        + b.y * vec2(v1.u, v1.v)
        + b.z * vec2(v2.u, v2.v);

    let d = cross(v1.p - v0.p, v2.p - v0.p);
    let area = length(d) / 2;
    if area == 0 {
        return ShapeSample();
    }
    var n_geo = normalize(d);

    if dot(n_geo, n_shade) < 0 {
        n_geo = -n_geo;
    }

    return ShapeSample(p, n_geo, uv, 1 / area);
}

fn triangle_pdf(tri: Triangle, ref_p: vec3f, p: vec3f) -> f32 {
    let v0 = TRI_VERTICES[tri.v0];
    let v1 = TRI_VERTICES[tri.v1];
    let v2 = TRI_VERTICES[tri.v2];

    let d = cross(v1.p - v0.p, v2.p - v0.p);
    let area = length(d) / 2;
    return 1 / area;
}
