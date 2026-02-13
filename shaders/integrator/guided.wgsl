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
    dir_node: u32,
    dir_node_2: u32,
    dir: vec2f,
    dir_filter_size: f32,
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

struct DirTreeNode {
    flux: f32,
    child: u32,
}

struct DirTreeNodeAtomic {
    flux: atomic<f32>,
    child: u32,
}

struct BoundingVolume {
    min: vec3f,
    max: vec3f,
}

@group(2) @binding(0)
var<storage, read_write> BSP_TREE: array<BspNode>;
@group(2) @binding(1)
var<storage> DIR_TREE_GUIDE: array<array<DirTreeNode, 4>>;
@group(2) @binding(2)
var<storage, read_write> DIR_TREE_TRAIN: array<array<DirTreeNodeAtomic, 4>>;
@group(2) @binding(3)
var<storage> BSP_VOLUME: BoundingVolume;

fn integrate_ray(wl: Wavelengths, ray_: Ray) -> vec4f {
    var radiance = vec4f();
    var throughput = vec4f(1);

    var path_vertices: array<PathVertex, MAX_LPV>;
    var pv_i = 0;

    var ray = ray_;

    var secondary_terminated = false;

    var depth = 0;
    while any(throughput > vec4f()) {
        let result = scene_raycast(ray, FLOAT_MAX);

        if !result.hit {
            // add infinite lights and finish
            for (var i = 1u; i < arrayLength(&INFINITE_LIGHTS); i++) {
                let emission = inf_light_emission(INFINITE_LIGHTS[i], ray, wl);
                radiance += throughput * emission;
                let power = dot(throughput, vec4f(1)) * dot(emission, vec4f(1));
                for (var j = 0; j < pv_i; j++) {
                    path_vertices[j].radiance += power / path_vertices[j].prefix_tp;
                }
            }
            break;
        }

        // add light emitted by surface
        {
            let emission = light_emission(result.light, ray, result, wl);
            radiance += throughput * emission;
            let power = dot(throughput, vec4f(1)) * dot(emission, vec4f(1));
            for (var j = 0; j < pv_i; j++) {
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
        if bsdf_is_highly_specular(bsdf) {
            pr_bsdf = 1;
        }

        var sample: BsdfSample;
        var filter_size: f32;

        let u = sample_1d();
        if u < pr_bsdf {
            // sample bsdf
            sample = bsdf_sample(bsdf, -ray.d, vec3f(sample_2d(), sample_1d()));
            if sample.pdf > 0 {
                var guide_pdf = vec2f();
                if !sample.specular && pr_bsdf < 1 {
                    guide_pdf = guide_pdf(guide, sample.dir);
                }
                sample.pdf = pr_bsdf * (sample.pdf + guide_pdf.x);
                filter_size = guide_pdf.y;
            }
        } else {
            // sample path guidance
            sample = guide_sample(guide, vec3f(sample_2d(), sample_1d()));
            if sample.pdf > 0 {
                filter_size = sample.f.x;
                sample.f = bsdf_f(bsdf, -ray.d, sample.dir);
                sample.pdf = (1 - pr_bsdf) * (sample.pdf + bsdf_pdf(bsdf, -ray.d, sample.dir));
            }
        }

        if sample.pdf == 0 {
            break;
        }

        throughput *= sample.f * abs(dot(bsdf_normal(bsdf), sample.dir)) / sample.pdf;

        if all(throughput == vec4f()) {
            break;
        }

        if !sample.specular {
            if pv_i == MAX_LPV {
                break;
            }
            let offset = vec3f(sample_2d(), sample_1d()) - 0.5;
            let alt_node = guide_locate(result.p + offset * spatial_node.filter_size);
            let alt_train = BSP_TREE[alt_node.node].right;
            let duv = equal_area_dir_to_square(sample.dir);
            path_vertices[pv_i] = PathVertex(train, alt_train, duv, filter_size, 0, dot(throughput, vec4f(1)));
            pv_i++;
        }

        // spawn new ray
        let offset = 10 * EPSILON * (1 + length(result.p));
        ray.d = sample.dir;
        ray.o = result.p + ray.d * offset;
    }

    for (var i = 0; i < pv_i; i++) {
        let v = path_vertices[i];
        guide_splat(v.dir_node, v.dir, v.radiance / 4);
        let offset_dir = v.dir + (sample_2d() - 0.5) * v.dir_filter_size;
        guide_splat(v.dir_node, wrap_equal_area_square(offset_dir), v.radiance / 4);

        guide_splat(v.dir_node_2, v.dir, v.radiance / 4);
        let offset_dir_2 = v.dir + (sample_2d() - 0.5) * v.dir_filter_size;
        guide_splat(v.dir_node_2, wrap_equal_area_square(offset_dir_2), v.radiance / 4);
    }

    return radiance;
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

    atomicAdd(&BSP_TREE[node].count, 1);

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
        let total_flux = children[0].flux
            + children[1].flux
            + children[2].flux
            + children[3].flux;
        if total_flux == 0.0 {
            break;
        }
        u *= total_flux;
        size *= 0.5;

        if u < children[0].flux + children[1].flux {
            if u < children[0].flux {
                u = u / children[0].flux;
                pdf *= 4 * children[0].flux / total_flux;
                node = children[0].child;
            } else {
                u = (u - children[0].flux)
                    / children[1].flux;
                pdf *= 4 * children[1].flux / total_flux;
                pos += vec2(size, 0);
                node = children[1].child;
            }
        } else {
            if u < children[0].flux + children[1].flux + children[2].flux {
                u = (u - children[0].flux - children[1].flux)
                    / children[2].flux;
                pdf *= 4 * children[2].flux / total_flux;
                pos += vec2(0, size);
                node = children[2].child;
            } else {
                u = (u - children[0].flux - children[1].flux - children[2].flux)
                    / children[3].flux;
                pdf *= 4 * children[3].flux / total_flux;
                pos += vec2(size, size);
                node = children[3].child;
            }
        }
    }

    let dir = equal_area_square_to_dir(random.xy * size + pos);

    return BsdfSample(vec4f(size), dir, pdf, false);
}

fn guide_pdf(dir_node: u32, dir: vec3f) -> vec2f {
    var pos = equal_area_dir_to_square(dir);
    var node = dir_node;
    var pdf = 1 / (2 * TWO_PI);
    var size = 1.0;
    while node != LEAF_SENTINEL {
        let children = DIR_TREE_GUIDE[node];
        let total = children[0].flux
            + children[1].flux
            + children[2].flux
            + children[3].flux;

        let child = u32(pos.x >= 0.5) + 2 * u32(pos.y >= 0.5);
        pdf *= 4 * children[child].flux / total;
        pos = fract(2 * pos);
        node = children[child].child;
        size *= 0.5;
    }
    return vec2f(pdf, size);
}

fn guide_splat(dir_node: u32, dir: vec2f, flux: f32) {
    var node = dir_node;
    var pos = dir;
    while node != LEAF_SENTINEL {
        let child = u32(pos.x >= 0.5) + 2 * u32(pos.y >= 0.5);
        atomicAdd(&DIR_TREE_TRAIN[node][child].flux, flux);
        pos = fract(2 * pos);
        node = DIR_TREE_TRAIN[node][child].child;
    }
}
