use bytemuck::{NoUninit, Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::{Transform, storage_buffer_entry};

#[derive(Default)]
pub struct Scene {
    pub spheres: Vec<Sphere>,
    pub triangles: Vec<Triangle>,

    pub triangle_vertices: Vec<TriVertex>,
}

const SHAPE_TAG_BITS: u32 = 1;
const SHAPE_TAG_SHIFT: u32 = 32 - SHAPE_TAG_BITS;
const SHAPE_IDX_MASK: u32 = (1 << SHAPE_TAG_SHIFT) - 1;
const SHAPE_TAG_MASK: u32 = !SHAPE_IDX_MASK;
const SHAPE_TAG_SPHERE: u32 = 0 << SHAPE_TAG_SHIFT;
const SHAPE_TAG_TRIANGLE: u32 = 1 << SHAPE_TAG_SHIFT;

impl Scene {
    pub fn make_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        let spheres = make_buffer(device, &self.spheres);
        let triangles = make_buffer(device, &self.triangles);

        let triangle_vertices = make_buffer(device, &self.triangle_vertices);

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene"),
            layout: &bind_group_layout(device),
            entries: &[
                make_entry(0, &spheres),
                make_entry(1, &triangles),
                make_entry(16, &triangle_vertices),
            ],
        })
    }

    pub fn add_sphere(&mut self, sphere: Sphere) -> u32 {
        let idx = self.spheres.len() as u32;
        self.spheres.push(sphere);
        idx | SHAPE_TAG_SPHERE
    }

    pub fn add_triangle_vertices(&mut self, positions: &[Vec3]) -> u32 {
        let base_index = self.triangle_vertices.len();
        self.triangle_vertices
            .extend(positions.iter().map(|&p| TriVertex { p, _padding: 0 }));
        base_index as u32
    }

    pub fn add_triangles(&mut self, indices: &[[u32; 3]]) -> u32 {
        let base_idx = self.triangles.len() as u32;
        self.triangles
            .extend(indices.iter().map(|&is| Triangle { vertices: is }));
        base_idx | SHAPE_TAG_TRIANGLE
    }
}

pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene"),
        entries: &[
            storage_buffer_entry(0),
            storage_buffer_entry(1),
            storage_buffer_entry(16),
        ],
    })
}

fn make_buffer<T: Pod>(device: &wgpu::Device, data: &[T]) -> wgpu::Buffer {
    let zeroed = T::zeroed();
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(std::any::type_name::<T>()),
        contents: match data.is_empty() {
            true => bytemuck::bytes_of(&zeroed),
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
pub struct Sphere {
    pub z_min: f32,
    pub z_max: f32,
    pub flip_normal: u32,
}

impl Sphere {
    pub const FULL: Sphere = Sphere {
        z_min: -1.0,
        z_max: 1.0,
        flip_normal: false as u32,
    };
}

#[derive(Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct Triangle {
    pub vertices: [u32; 3],
}

#[derive(Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct TriVertex {
    pub p: Vec3,
    pub _padding: u32,
}
