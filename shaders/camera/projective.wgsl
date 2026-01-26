#import /transform.wgsl
#import /util/distr.wgsl

@group(1) @binding(16)
var<storage, read> camera_data: ProjectiveCamera;

struct ProjectiveCamera {
    ndc_to_camera: Transform,
    world_to_camera: Transform,
    lens_radius: f32,
    focal_distance: f32,
    orthographic: u32,
}

fn camera_sample_ray(film_ndc: vec2f) -> Ray {
    let projected = transform_point(camera_data.ndc_to_camera, vec3(film_ndc, 0));
    let time = sample_1d();
    var ray: Ray;
    if camera_data.orthographic != 0 {
        ray = Ray(projected, vec3f(0, 0, 1), time);
    } else {
        ray = Ray(vec3f(), normalize(projected), time);
    }

    if camera_data.lens_radius > 0 {
        let lens_sample = camera_data.lens_radius * sample_uniform_disk(sample_2d());

        let focal_t = camera_data.focal_distance / ray.d.z;
        let focal_p = ray.o + ray.d * focal_t;

        // note: pbr-book simply sets the ray origin to the lens position instead of offsetting it.
        //       IMO this doesn't make any sense for orthographic projections: as the lens
        //       radius goes to zero, this approaches a pinhole perspective projection.
        //       So instead, I offset the ray origin by the lens radius: as the lens radius
        //       goes to zero, this approaches a pinhole orthographic projection.
        //       In fact, the if statement is unnecessary as when lens_radius is zero the ray is
        //       unaffected.
        ray.o += vec3(lens_sample, 0);
        ray.d = normalize(focal_p - ray.o);
    }

    return transform_ray_inv(camera_data.world_to_camera, ray);
}
