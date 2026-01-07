#import /ray.wgsl

struct Transform {
    m: mat4x4f,
    m_inv: mat4x4f,
}

fn transform_point(transform: Transform, p: vec3f) -> vec3f {
    let new_p = transform.m * vec4(p, 1);
    if new_p.w == 1 {
        return new_p.xyz;
    } else {
        return new_p.xyz / new_p.w;
    }
}

fn transform_vector(transform: Transform, v: vec3f) -> vec3f {
    return (transform.m * vec4(v, 0)).xyz;
}

fn transform_normal(transform: Transform, n: vec3f) -> vec3f {
    return (transpose(transform.m_inv) * vec4(n, 0)).xyz;
}

fn transform_ray(transform: Transform, ray: Ray) -> Ray {
    return Ray(
        transform_point(transform, ray.o),
        transform_vector(transform, ray.d),
        ray.time,
    );
}


fn transform_point_inv(transform: Transform, p: vec3f) -> vec3f {
    return transform_point(Transform(transform.m_inv, transform.m), p);
}

fn transform_vector_inv(transform: Transform, v: vec3f) -> vec3f {
    return transform_vector(Transform(transform.m_inv, transform.m), v);
}

fn transform_normal_inv(transform: Transform, n: vec3f) -> vec3f {
    return transform_normal(Transform(transform.m_inv, transform.m), n);
}

fn transform_ray_inv(transform: Transform, ray: Ray) -> Ray {
    return transform_ray(Transform(transform.m_inv, transform.m), ray);
}
