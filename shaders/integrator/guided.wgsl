#import /scene.wgsl
#import /ray.wgsl
#import /util/misc.wgsl
#import /material.wgsl
#import /light.wgsl
#import /light_sampler.wgsl

const MAX_DEPTH = 250;
const MAX_LPV = 10;
const PR_BSDF: f32 = 0.5;

struct PathVertex {
    pos: vec3f,
    dir: vec2f,
    pos_filter_size: f32,
    radiance: f32,
    prefix_tp: f32,
}

const LEAF_SENTINEL: u32 = ~0u;

struct BspNode {
    is_leaf: u32,
    left: u32,
    right: u32,
    count: atomic<u32>,
}

struct GuideNode {
    child: u32,
    pr: f32,
}

struct TrainNode {
    child: u32,
    sum: atomic<f32>,
    comp: atomic<f32>,
}

struct BoundingVolume {
    min: vec3f,
    max: vec3f,
}

@group(2) @binding(0)
var<storage, read_write> BSP_TREE: array<BspNode>;
@group(2) @binding(1)
var<storage> DIR_TREE_GUIDE: array<array<GuideNode, 4>>;
@group(2) @binding(2)
var<storage, read_write> DIR_TREE_TRAIN: array<array<TrainNode, 4>>;
@group(2) @binding(3)
var<storage> BSP_VOLUME: BoundingVolume;

const POS_STRAT = array(
    vec3f(0, 0, 0),
    vec3f(0.25, 0.75, 0.5),
    vec3f(0.5, 0.25, 0.75),
    vec3f(0.75, 0.5, 0.25),
);
const DIR_STRAT = array(
    vec2f(0, 0),
    vec2f(0.25, 0.5),
    vec2f(0.5, 0.5),
    vec2f(0.75, 0.25),
);

fn integrate_ray(wl: Wavelengths, ray_: Ray) -> vec4f {
    var radiance = vec4f();
    var throughput = vec4f(1);

    var path_vertices: array<PathVertex, MAX_LPV>;
    var pv_i = 0;

    var ray = ray_;

    var secondary_terminated = false;
    var specular_bounce = false;
    var bounce_pdf: f32;

    var depth = 0;
    while any(throughput > vec4f()) {
        let result = scene_raycast(ray, FLOAT_MAX);

        if !result.hit {
            // add infinite lights and finish
            if depth == 0 || specular_bounce {
                for (var i = 1u; i < arrayLength(&INFINITE_LIGHTS); i++) {
                    let emission = inf_light_emission(INFINITE_LIGHTS[i], ray, wl);
                    radiance += throughput * emission;
                    let power = dot(throughput, vec4f(1)) * dot(emission, vec4f(1));
                    for (var j = 0; j < pv_i; j++) {
                        path_vertices[j].radiance += power / path_vertices[j].prefix_tp;
                    }
                }
            } else {
                for (var i = 1u; i < arrayLength(&INFINITE_LIGHTS); i++) {
                    let ls_pdf = light_sampler_pmf(ROOT_LS, ray.o, INFINITE_LIGHTS[i])
                        * light_pdf(INFINITE_LIGHTS[i], ray.o, ray.d);
                    let emission = inf_light_emission(INFINITE_LIGHTS[i], ray, wl)
                        * ls_mis_weight(bounce_pdf, ls_pdf);

                    radiance += throughput * emission;
                    let power = dot(throughput, vec4f(1)) * dot(emission, vec4f(1));
                    for (var j = 0; j < pv_i - 1; j++) {
                        path_vertices[j].radiance += power / path_vertices[j].prefix_tp;
                    }
                }
            }
            break;
        }

        // add light emitted by surface
        if depth == 0 || specular_bounce {
            let emission = light_emission(result.light, ray, result, wl);

            radiance += throughput * emission;
            let power = dot(throughput, vec4f(1)) * dot(emission, vec4f(1));
            for (var j = 0; j < pv_i; j++) {
                path_vertices[j].radiance += power / path_vertices[j].prefix_tp;
            }
        } else {
            let ls_pdf = light_sampler_pmf(ROOT_LS, ray.o, result.light)
                * light_pdf(result.light, ray.o, ray.d);
            let emission = light_emission(result.light, ray, result, wl)
                * ls_mis_weight(bounce_pdf, ls_pdf);

            radiance += throughput * emission;
            let power = dot(throughput, vec4f(1)) * dot(emission, vec4f(1));
            for (var j = 0; j < pv_i - 1; j++) {
                path_vertices[j].radiance += power / path_vertices[j].prefix_tp;
            }
        }

        // enforce termination
        depth += 1;
        if depth > MAX_DEPTH {
            break;
        }

        let spatial_node = guide_locate(result.p);
        let guide = BSP_TREE[spatial_node.node].left;
        let train = BSP_TREE[spatial_node.node].right;

        let bsdf = material_evaluate(result.material, result, wl);

        if !secondary_terminated && bsdf_terminates_secondary_wavelengths(bsdf) {
            secondary_terminated = true;
            throughput *= vec4f(4, 0, 0, 0);
        }

        var pr_bsdf = PR_BSDF;
        if bsdf_is_highly_specular(bsdf) || guide == LEAF_SENTINEL {
            pr_bsdf = 1;
        }

        // sample direct lighting
        {
            let contribution = _sample_direct_light(bsdf, guide, pr_bsdf, result, ray, wl);
            radiance += throughput * contribution;
            let power = dot(throughput, vec4f(1)) * dot(contribution, vec4f(1));
            for (var j = 0; j < pv_i; j++) {
                path_vertices[j].radiance += power / path_vertices[j].prefix_tp;
            }
        }

        var sample: BsdfSample;

        let u = vec3f(sample_2d(), sample_1d());
        if u.z < pr_bsdf {
            // sample bsdf
            sample = bsdf_sample(bsdf, -ray.d, vec3f(u.xy, u.z / pr_bsdf));
            if sample.pdf > 0 {
                var guide_pdf = 0.0;
                if !sample.specular && pr_bsdf < 1 {
                    guide_pdf = guide_pdf(guide, sample.dir);
                }
                sample.pdf = pr_bsdf * (sample.pdf + guide_pdf);
            }
        } else {
            // sample path guidance
            sample = guide_sample(guide, vec3f(u.xy, (u.z - pr_bsdf) / (1 - pr_bsdf)));
            if sample.pdf > 0 {
                sample.f = bsdf_f(bsdf, -ray.d, sample.dir);
                sample.pdf = (1 - pr_bsdf) * (sample.pdf + bsdf_pdf(bsdf, -ray.d, sample.dir));
            }
        }

        if sample.pdf == 0 {
            break;
        }

        throughput *= sample.f * abs(dot(bsdf_normal(bsdf), sample.dir)) / sample.pdf;
        bounce_pdf = sample.pdf;

        if all(throughput == vec4f()) {
            break;
        }

        if !sample.specular {
            if pv_i == MAX_LPV {
                break;
            }
            let duv = equal_area_dir_to_square(sample.dir);
            path_vertices[pv_i] = PathVertex(
                result.p,
                duv,
                dot(spatial_node.filter_size, vec3f(1)) / 3.0,
                0,
                dot(throughput, vec4f(1))
            );
            pv_i++;
        }

        // spawn new ray
        let offset = 10 * EPSILON * (1 + length(result.p));
        ray.d = sample.dir;
        ray.o = result.p + ray.d * offset;
        specular_bounce = sample.specular;
    }

    for (var i = 0; i < pv_i; i++) {
        let v = path_vertices[i];
        let pos_jitter = vec3f(sample_2d(), sample_1d());
        for (var j = 0; j < 4; j++) {
            let node = guide_locate(v.pos + (fract(pos_jitter + POS_STRAT[j]) - 0.5) * v.pos_filter_size).node;
            atomicAdd(&BSP_TREE[node].count, 1);
            let dir_node = BSP_TREE[node].right;
            let dir_jitter = sample_2d();
            let dir_filter_size = guide_filter_size(dir_node, v.dir);
            for (var k = 0; k < 4; k++) {
                let offset_dir = v.dir + (fract(dir_jitter + DIR_STRAT[k]) - 0.5) * dir_filter_size;
                guide_splat(dir_node, wrap_equal_area_square(offset_dir), v.radiance / 4);
            }
        }
    }

    return radiance;
}

fn _sample_direct_light(
    bsdf: Bsdf,
    dir_node: u32,
    pr_bsdf: f32,
    hit: RaycastResult,
    ray_: Ray,
    wl: Wavelengths,
) -> vec4f {
    let light_id_sample = light_sampler_sample(ROOT_LS, hit.p, sample_1d());
    if light_id_sample.pmf == 0 {
        return vec4f();
    }

    let light_sample = light_sample(light_id_sample.light, hit.p, wl, sample_2d());
    if light_sample.pdf_wrt_solid_angle == 0 {
        return vec4f();
    }

    let pdf = light_sample.pdf_wrt_solid_angle * light_id_sample.pmf;

    let bounce_pdf = mix(
        guide_pdf(dir_node, light_sample.dir),
        bsdf_pdf(bsdf, -ray_.d, light_sample.dir),
        pr_bsdf,
    );

    let contribution = light_sample.emission
        * bsdf_f(bsdf, -ray_.d, light_sample.dir)
        * abs(dot(bsdf_normal(bsdf), light_sample.dir))
        / pdf
        * ls_mis_weight(pdf, bounce_pdf);

    if all(contribution == vec4f()) {
        return vec4f();
    }

    var ray = ray_;
    let offset = 10 * EPSILON * (1 + length(hit.p));
    ray.d = light_sample.dir;
    ray.o = hit.p + ray.d * offset;

    if scene_raycast(ray, light_sample.t_max - offset - 0.0001).hit {
        return vec4f();
    }

    return contribution;
}

struct SpatialInfo {
    node: u32,
    filter_size: vec3f,
}

fn guide_locate(p_: vec3f) -> SpatialInfo {
    var size = BSP_VOLUME.max - BSP_VOLUME.min;
    var p = (p_ - BSP_VOLUME.min) / size;
    p = clamp(p, vec3f(0), vec3f(1));

    var node = 0u;
    var axis = 0;
    while BSP_TREE[node].is_leaf == 0 {
        if p[axis] < 0.5 {
            node = BSP_TREE[node].left;
        } else {
            node = BSP_TREE[node].right;
            p[axis] -= 0.5;
        }
        p[axis] *= 2;
        size[axis] *= 0.5;
        axis = (axis + 1) % 3;
    }

    return SpatialInfo(node, size);
}

fn guide_sample(dir_node: u32, random: vec3f) -> BsdfSample {
    var u = random.z;
    var node = dir_node;
    var pos = vec2f();
    var size = 1.0;
    var pdf = 1 / (2 * TWO_PI);
    while node != LEAF_SENTINEL {
        let children = DIR_TREE_GUIDE[node];
        size *= 0.5;

        if u < children[0].pr + children[1].pr {
            if u < children[0].pr {
                u = u / children[0].pr;
                pdf *= 4 * children[0].pr;
                node = children[0].child;
            } else {
                u = (u - children[0].pr)
                    / children[1].pr;
                pdf *= 4 * children[1].pr;
                pos += vec2(size, 0);
                node = children[1].child;
            }
        } else {
            if u < children[0].pr + children[1].pr + children[2].pr {
                u = (u - children[0].pr - children[1].pr)
                    / children[2].pr;
                pdf *= 4 * children[2].pr;
                pos += vec2(0, size);
                node = children[2].child;
            } else {
                u = (u - children[0].pr - children[1].pr - children[2].pr)
                    / children[3].pr;
                pdf *= 4 * children[3].pr;
                pos += vec2(size, size);
                node = children[3].child;
            }
        }
    }

    let dir = equal_area_square_to_dir(random.xy * size + pos);

    return BsdfSample(vec4f(), dir, pdf, false);
}

fn guide_pdf(dir_node: u32, dir: vec3f) -> f32 {
    var pos = equal_area_dir_to_square(dir);
    var node = dir_node;
    var pdf = 1 / (2 * TWO_PI);
    while node != LEAF_SENTINEL {
        let children = DIR_TREE_GUIDE[node];
        let child = u32(pos.x >= 0.5) + 2 * u32(pos.y >= 0.5);
        pdf *= 4 * children[child].pr;
        pos = fract(2 * pos);
        node = children[child].child;
    }
    return pdf;
}

fn guide_filter_size(dir_node: u32, dir: vec2f) -> f32 {
    var size = 1.0;
    var node = dir_node;
    var pos = dir;
    while node != LEAF_SENTINEL {
        let child = u32(pos.x >= 0.5) + 2 * u32(pos.y >= 0.5);
        pos = fract(2 * pos);
        node = DIR_TREE_TRAIN[node][child].child;
        size *= 0.5;
    }
    return size;
}

fn guide_splat(dir_node: u32, dir: vec2f, value: f32) {
    var node = dir_node;
    var pos = dir;
    loop {
        let child = &DIR_TREE_TRAIN[node][u32(pos.x >= 0.5) + 2 * u32(pos.y >= 0.5)];
        if child.child == LEAF_SENTINEL {
            // Kahan-Babuska-Neumaier summation
            let old_sum = atomicAdd(&child.sum, value);
            let new_sum = old_sum + value;
            var lost: f32;
            // no abs() because all values are positive
            if old_sum >= value {
                lost = (old_sum - new_sum) + value;
            } else {
                lost = (value - new_sum) + old_sum;
            }
            atomicAdd(&child.comp, lost);
            break;
        }
        pos = fract(2 * pos);
        node = child.child;
    }
}

fn ls_mis_weight(p1: f32, p2: f32) -> f32 {
    return p1 / (p1 + p2);
}
