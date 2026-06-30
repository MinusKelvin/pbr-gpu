use std::ops::Range;
use std::time::Instant;

use bytemuck::NoUninit;
use glam::Vec3;
use rayon::prelude::*;

use crate::Transform;
use crate::scene::{Bounds, Scene, ShapeId, ShapeType};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, NoUninit)]
#[repr(C)]
pub struct NodeId(u32);

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
enum NodeType {
    Bvh = 0 << NodeId::TAG_SHIFT,
    Transform = 1 << NodeId::TAG_SHIFT,
    Primitive = 2 << NodeId::TAG_SHIFT,
}

#[derive(Default)]
pub struct AccelerationStructureBuild {
    pub bvh_nodes: Vec<BvhNode>,
    pub transform_nodes: Vec<TransformNode>,
    pub triangle_nodes: Vec<TriangleNode>,
}

#[allow(unused)]
impl NodeId {
    pub const ZERO: NodeId = NodeId(0);

    const TAG_BITS: u32 = 2;
    const TAG_SHIFT: u32 = 32 - Self::TAG_BITS;
    const IDX_MASK: u32 = (1 << Self::TAG_SHIFT) - 1;
    const TAG_MASK: u32 = !Self::IDX_MASK;

    fn new(ty: NodeType, idx: usize) -> Self {
        assert!(
            idx <= Self::IDX_MASK as usize,
            "cannot exceed {} {ty:?} shapes",
            Self::IDX_MASK
        );
        NodeId(idx as u32 | ty as u32)
    }

    fn ty(self) -> NodeType {
        unsafe { std::mem::transmute(self.0 & Self::TAG_MASK) }
    }

    fn idx(self) -> usize {
        (self.0 & Self::IDX_MASK) as usize
    }
}

impl AccelerationStructureBuild {
    pub fn create_blas(&mut self, tri_groups: &[Range<usize>], scene: &Scene) -> NodeId {
        let nodes = tri_groups
            .iter()
            .cloned()
            .flatten()
            .map(|id| self.add_triangle_node(id as u32))
            .collect::<Vec<_>>();
        self.add_bvh(&nodes, scene)
    }

    pub fn create_tlas(
        &mut self,
        instances: impl Iterator<Item = (Transform, NodeId)>,
        scene: &Scene,
    ) -> NodeId {
        let nodes: Vec<_> = instances
            .map(|(tform, node)| self.add_transform(tform, node))
            .collect();
        self.add_bvh(&nodes, scene)
    }

    fn add_triangle_node(&mut self, triangle: u32) -> NodeId {
        let id = NodeId::new(NodeType::Primitive, self.triangle_nodes.len());
        self.triangle_nodes.push(TriangleNode {
            tri_index: triangle,
        });
        id
    }

    fn add_transform(&mut self, transform: Transform, node: NodeId) -> NodeId {
        let id = NodeId::new(NodeType::Transform, self.transform_nodes.len());
        self.transform_nodes.push(TransformNode {
            transform,
            object: node,
            _padding: [0; 3],
        });
        id
    }

    fn add_bvh(&mut self, nodes: &[NodeId], scene: &Scene) -> NodeId {
        let t = Instant::now();

        let mut bounded_objects: Vec<_> = nodes
            .iter()
            .map(|&id| (id, self.node_bounds(id, scene)))
            .collect();
        let result = self.build_bvh(&mut bounded_objects);

        println!("Build BVH in {:.3?}", t.elapsed());

        result
    }

    fn build_bvh(&mut self, objs: &mut [(NodeId, Bounds)]) -> NodeId {
        assert!(!objs.is_empty());

        let idx = self.bvh_nodes.len();
        self.bvh_nodes.push(BvhNode {
            min: Vec3::ZERO,
            flags: 0,
            max: Vec3::ZERO,
            far_node: NodeId(0),
        });

        if let &mut [(node, ref bounds)] = objs {
            self.bvh_nodes[idx].min = bounds.min;
            self.bvh_nodes[idx].max = bounds.max;
            self.bvh_nodes[idx].far_node = node;
            self.bvh_nodes[idx].flags = 0;
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
            assert_eq!(idx + 1, left_node.idx());
            let right_node = self.build_bvh(right);

            self.bvh_nodes[idx].min = total_bounds.min;
            self.bvh_nodes[idx].max = total_bounds.max;
            self.bvh_nodes[idx].far_node = right_node;
            self.bvh_nodes[idx].flags = 1 << axis;
        }

        NodeId::new(NodeType::Bvh, idx)
    }

    pub fn node_bounds(&self, node: NodeId, scene: &Scene) -> Bounds {
        match node.ty() {
            NodeType::Primitive => scene.shape_bounds(ShapeId::new(
                ShapeType::Triangle,
                self.triangle_nodes[node.idx()].tri_index as usize,
            )),
            NodeType::Bvh => {
                let bvh = &self.bvh_nodes[node.idx()];
                Bounds {
                    min: bvh.min,
                    max: bvh.max,
                }
            }
            NodeType::Transform => {
                let node = &self.transform_nodes[node.idx()];
                let bounds = self.node_bounds(node.object, scene);
                Bounds::from_points(
                    bounds
                        .corners()
                        .into_iter()
                        .map(|p| node.transform.m_inv.transform_point3(p)),
                )
            }
        }
    }
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct BvhNode {
    pub min: Vec3,
    pub flags: u32,
    pub max: Vec3,
    pub far_node: NodeId,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct TransformNode {
    pub transform: Transform,
    pub object: NodeId,
    pub _padding: [u32; 3],
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct TriangleNode {
    pub tri_index: u32,
}
