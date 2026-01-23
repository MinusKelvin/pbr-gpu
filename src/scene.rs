use bytemuck::NoUninit;
use glam::{BVec3, Vec3};
use wgpu::util::DeviceExt;

use crate::storage_buffer_entry;

mod node;
mod shapes;

pub use self::node::*;
pub use self::shapes::*;

#[derive(Default)]
pub struct Scene {
    pub spheres: Vec<Sphere>,
    pub triangles: Vec<Triangle>,

    pub triangle_vertices: Vec<TriVertex>,

    pub bvh_nodes: Vec<BvhNode>,
    pub transform_nodes: Vec<TransformNode>,
    pub primitive_nodes: Vec<PrimitiveNode>,

    pub root: Option<NodeId>,
}

impl Scene {
    pub fn make_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        let spheres = make_buffer(device, &self.spheres);
        let triangles = make_buffer(device, &self.triangles);

        let triangle_vertices = make_buffer(device, &self.triangle_vertices);

        let bvh = make_buffer(device, &self.bvh_nodes);
        let transform = make_buffer(device, &self.transform_nodes);
        let primitive = make_buffer(device, &self.primitive_nodes);

        let root = make_buffer(device, &[self.root.unwrap()]);

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
