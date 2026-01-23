use bytemuck::{NoUninit, Pod, Zeroable};
use glam::Vec3;

use crate::scene::{Bounds, Scene};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, NoUninit)]
#[repr(C)]
pub struct ShapeId(u32);

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
enum ShapeType {
    Sphere = 0 << ShapeId::TAG_SHIFT,
    Triangle = 1 << ShapeId::TAG_SHIFT,
}

impl ShapeId {
    const TAG_BITS: u32 = 1;
    const TAG_SHIFT: u32 = 32 - Self::TAG_BITS;
    const IDX_MASK: u32 = (1 << Self::TAG_SHIFT) - 1;
    const TAG_MASK: u32 = !Self::IDX_MASK;

    fn new(ty: ShapeType, idx: usize) -> Self {
        assert!(
            idx <= Self::IDX_MASK as usize,
            "cannot exceed {} {ty:?} shapes",
            Self::IDX_MASK
        );
        ShapeId(idx as u32 | ty as u32)
    }

    fn ty(self) -> ShapeType {
        unsafe { std::mem::transmute(self.0 & Self::TAG_MASK) }
    }

    fn idx(self) -> usize {
        (self.0 & Self::IDX_MASK) as usize
    }
}

impl Scene {
    pub fn shape_bounds(&self, shape: ShapeId) -> Bounds {
        match shape.ty() {
            ShapeType::Sphere => self.spheres[shape.idx()].bounds(),
            ShapeType::Triangle => self.triangles[shape.idx()].bounds(&self.triangle_vertices),
        }
    }

    pub fn add_sphere(&mut self, sphere: Sphere) -> ShapeId {
        let id = ShapeId::new(ShapeType::Sphere, self.spheres.len());
        self.spheres.push(sphere);
        id
    }

    pub fn add_triangles(
        &mut self,
        verts: &[TriVertex],
        tris: &[[u32; 3]],
    ) -> impl Iterator<Item = ShapeId> + use<> {
        let base_index = self.triangle_vertices.len();
        self.triangle_vertices.extend(verts);

        let base_idx = self.triangles.len();
        self.triangles.extend(tris.iter().map(|idx| Triangle {
            vertices: idx.map(|i| i + base_index as u32),
        }));
        let end_idx = self.triangles.len();

        (base_idx..end_idx).map(|idx| ShapeId::new(ShapeType::Triangle, idx))
    }
}

#[derive(Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct Sphere {
    pub z_min: f32,
    pub z_max: f32,
    pub flip_normal: u32,
}

impl Sphere {
    fn bounds(&self) -> Bounds {
        Bounds {
            min: Vec3::new(-1.0, -1.0, self.z_min),
            max: Vec3::new(1.0, 1.0, self.z_max),
        }
    }
}

#[derive(Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct TriVertex {
    pub p: Vec3,
    pub u: f32,
    pub n: Vec3,
    pub v: f32,
}

#[derive(Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct Triangle {
    pub vertices: [u32; 3],
}

impl Triangle {
    fn bounds(&self, verts: &[TriVertex]) -> Bounds {
        Bounds::from_points(self.vertices.iter().map(|&id| verts[id as usize].p))
    }
}
