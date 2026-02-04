#import /util/misc.wgsl
#import /ray.wgsl

struct Sphere {
    z_min: f32,
    z_max: f32,
    flip_normal: u32,
}

struct _SphereHit {
    hit: bool,
    t: f32,
    p: vec3f,
}

fn _sphere_test_t(sphere: Sphere, ray: Ray, t_max: f32, t: f32) -> _SphereHit {
    if t <= 0 || t > t_max {
        return _SphereHit();
    }

    let p = normalize(ray.o + ray.d * t);

    if sphere.z_min > -1 && p.z < sphere.z_min
        || sphere.z_max < 1 && p.z > sphere.z_max {
        return _SphereHit();
    }

    return _SphereHit(true, t, p);
}

fn _sphere_hit(sphere: Sphere, ray: Ray, t_max: f32) -> _SphereHit {
    let a = dot(ray.d, ray.d);
    let b = dot(ray.o, ray.d) * 2;
    let c = dot(ray.o, ray.o) - 1;

    // more accurate version of b^2 - 4ac (via pbr-book.org)
    let v = ray.o - b / (2 * a) * ray.d;
    let l = length(v);
    let discrim = 4 * a * (1 + l) * (1 - l);

    if discrim < 0 {
        return _SphereHit();
    }

    let root = sqrt(discrim);
    let q = -0.5 * (b + copysign(root, b));
    var t0 = q / a;
    var t1 = c / q;

    if t0 > t1 {
        let tmp = t0;
        t0 = t1;
        t1 = tmp;
    }

    var result = _sphere_test_t(sphere, ray, t_max, t0);
    if !result.hit {
        result = _sphere_test_t(sphere, ray, t_max, t1);
    }

    return result;
}

fn sphere_raycast(sphere: Sphere, ray: Ray, t_max: f32) -> RaycastResult {
    let hit = _sphere_hit(sphere, ray, t_max);
    if !hit.hit {
        return RaycastResult();
    }
    let n = select(hit.p, -hit.p, sphere.flip_normal != 0u);
    return RaycastResult(
        true,
        hit.p,
        n,
        n,
        hit.t,
        MaterialId(),
        LightId(),
        vec2f(),
    );
}

fn sphere_sample(sphere: Sphere, p: vec3f, random: vec2f) -> ShapeSample {
    return ShapeSample();
}
