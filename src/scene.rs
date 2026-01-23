use std::time::Instant;

use bytemuck::{NoUninit, Pod, Zeroable};
use glam::{BVec3, Vec3};
use rayon::prelude::ParallelSliceMut;
use wgpu::util::DeviceExt;

pub use crate::scene::shapes::*;
use crate::{Transform, storage_buffer_entry};

mod shapes;

#[derive(Default)]
pub struct Scene {
    pub spheres: Vec<Sphere>,
    pub triangles: Vec<Triangle>,

    pub triangle_vertices: Vec<TriVertex>,

    pub bvh_nodes: Vec<BvhNode>,
    pub transform_nodes: Vec<TransformNode>,
    pub primitive_nodes: Vec<PrimitiveNode>,

    pub root: u32,
}

const NODE_TAG_BITS: u32 = 2;
const NODE_TAG_SHIFT: u32 = 32 - NODE_TAG_BITS;
const NODE_IDX_MASK: u32 = (1 << NODE_TAG_SHIFT) - 1;
const NODE_TAG_MASK: u32 = !NODE_IDX_MASK;
const NODE_TAG_BVH: u32 = 0 << NODE_TAG_SHIFT;
const NODE_TAG_TRANSFORM: u32 = 1 << NODE_TAG_SHIFT;
const NODE_TAG_PRIMITIVE: u32 = 2 << NODE_TAG_SHIFT;

impl Scene {
    pub fn make_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        let spheres = make_buffer(device, &self.spheres);
        let triangles = make_buffer(device, &self.triangles);

        let triangle_vertices = make_buffer(device, &self.triangle_vertices);

        let bvh = make_buffer(device, &self.bvh_nodes);
        let transform = make_buffer(device, &self.transform_nodes);
        let primitive = make_buffer(device, &self.primitive_nodes);

        let root = make_buffer(device, &self.root.to_le_bytes());

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene"),
            layout: &bind_group_layout(device),
            entries: &[
                make_entry(0, &spheres),
                make_entry(1, &triangles),
                make_entry(16, &triangle_vertices),
                make_entry(32, &root),
                make_entry(33, &bvh),
                make_entry(34, &transform),
                make_entry(35, &primitive),
            ],
        })
    }

    pub fn add_primitive(&mut self, prim: PrimitiveNode) -> u32 {
        let idx = self.primitive_nodes.len() as u32;
        self.primitive_nodes.push(prim);
        idx | NODE_TAG_PRIMITIVE
    }

    pub fn add_bvh(&mut self, nodes: &[u32]) -> u32 {
        let t = Instant::now();

        let mut bounded_objects: Vec<_> =
            nodes.iter().map(|&id| (id, self.node_bounds(id))).collect();
        let result = self.build_bvh(&mut bounded_objects);

        eprintln!("Build BVH in {:.3?}", t.elapsed());

        result
    }

    pub fn add_transform(&mut self, transform: Transform, node: u32) -> u32 {
        let idx = self.transform_nodes.len() as u32;
        self.transform_nodes.push(TransformNode {
            transform,
            object: node,
            _padding: [0; 3],
        });
        idx | NODE_TAG_TRANSFORM
    }

    fn build_bvh(&mut self, objs: &mut [(u32, Bounds)]) -> u32 {
        assert!(!objs.is_empty());

        let id = self.bvh_nodes.len();
        self.bvh_nodes.push(BvhNode {
            min: Vec3::ZERO,
            flags: 0,
            max: Vec3::ZERO,
            far_node: 0,
        });

        if let &mut [(node, ref bounds)] = objs {
            self.bvh_nodes[id].min = bounds.min;
            self.bvh_nodes[id].max = bounds.max;
            self.bvh_nodes[id].far_node = node;
            self.bvh_nodes[id].flags = 0;
        } else {
            let total_bounds = objs
                .iter()
                .fold(objs[0].1.clone(), |acc, (_, bb)| acc.union(bb));

            let axis = total_bounds.size().max_position();
            objs.par_sort_unstable_by_key(|(_, bb)| {
                ordered_float::OrderedFloat(bb.centroid()[axis])
            });

            let mut costs = vec![0.0; objs.len() - 1];

            let mut bb = objs[0].1.clone();
            for i in 1..objs.len() {
                costs[i - 1] += i as f32 * bb.surface_area();
                bb = bb.union(&objs[i].1);
            }

            let mut bb = objs.last().unwrap().1.clone();
            for i in 1..objs.len() {
                costs[objs.len() - 1 - i] += i as f32 * bb.surface_area();
                bb = bb.union(&objs[objs.len() - 1 - i].1);
            }

            let split = 1 + costs
                .iter()
                .enumerate()
                .min_by_key(|&(_, &cost)| ordered_float::OrderedFloat(cost))
                .unwrap()
                .0;

            let (left, right) = objs.split_at_mut(split);

            let left_node = self.build_bvh(left);
            assert_eq!(id as u32 + 1, left_node & NODE_IDX_MASK);
            let right_node = self.build_bvh(right);

            self.bvh_nodes[id].min = total_bounds.min;
            self.bvh_nodes[id].max = total_bounds.max;
            self.bvh_nodes[id].far_node = right_node;
            self.bvh_nodes[id].flags = 1 << axis;
        }

        id as u32 | NODE_TAG_BVH
    }

    fn node_bounds(&self, node: u32) -> Bounds {
        match node & NODE_TAG_MASK {
            NODE_TAG_PRIMITIVE => {
                self.shape_bounds(self.primitive_nodes[(node & NODE_IDX_MASK) as usize].shape)
            }
            NODE_TAG_BVH => {
                let bvh = &self.bvh_nodes[(node & NODE_IDX_MASK) as usize];
                Bounds {
                    min: bvh.min,
                    max: bvh.max,
                }
            }
            NODE_TAG_TRANSFORM => {
                let node = &self.transform_nodes[(node & NODE_IDX_MASK) as usize];
                let bounds = self.node_bounds(node.object);
                Bounds::from_points(
                    bounds
                        .corners()
                        .into_iter()
                        .map(|p| node.transform.m_inv.transform_point3(p)),
                )
            }
            _ => unreachable!(),
        }
    }
}

pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene"),
        entries: &[
            storage_buffer_entry(0),
            storage_buffer_entry(1),
            storage_buffer_entry(16),
            storage_buffer_entry(32),
            storage_buffer_entry(33),
            storage_buffer_entry(34),
            storage_buffer_entry(35),
        ],
    })
}

fn make_buffer<T: NoUninit>(device: &wgpu::Device, data: &[T]) -> wgpu::Buffer {
    let empty = vec![0; std::mem::size_of::<T>()];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(std::any::type_name::<T>()),
        contents: match data.is_empty() {
            true => &empty,
            false => bytemuck::cast_slice(data),
        },
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn make_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[derive(Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct BvhNode {
    pub min: Vec3,
    pub flags: u32,
    pub max: Vec3,
    pub far_node: u32,
}

const BVH_FLAG_X: u32 = 1 << 0;
const BVH_FLAG_Y: u32 = 1 << 1;
const BVH_FLAG_Z: u32 = 1 << 2;

#[derive(Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct TransformNode {
    pub transform: Transform,
    pub object: u32,
    pub _padding: [u32; 3],
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct PrimitiveNode {
    pub shape: ShapeId,
}

#[derive(Clone, Debug)]
pub struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl Bounds {
    fn from_points(mut points: impl Iterator<Item = Vec3>) -> Self {
        let first = points.next().unwrap();
        let mut this = Bounds {
            min: first,
            max: first,
        };
        for p in points {
            this.min = this.min.min(p);
            this.max = this.max.max(p);
        }
        this
    }

    fn surface_area(&self) -> f32 {
        let size = self.size();
        2.0 * (size.x * size.y + size.x * size.z + size.y * size.z)
    }

    fn union(&self, other: &Bounds) -> Bounds {
        Bounds {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn size(&self) -> Vec3 {
        self.max - self.min
    }

    fn centroid(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    fn corners(&self) -> [Vec3; 8] {
        [0, 1, 2, 3, 4, 5, 6, 7].map(|i| {
            Vec3::select(
                BVec3::new(i & 1 != 0, i & 2 != 0, i & 4 != 0),
                self.max,
                self.min,
            )
        })
    }
}
