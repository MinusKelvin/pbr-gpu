use bytemuck::{NoUninit, Pod, Zeroable};
use wgpu::util::DeviceExt;

pub struct Scene {
    pub spheres: Vec<Sphere>,
}

impl Scene {
    pub fn make_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        let spheres = make_buffer(device, &self.spheres);

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene"),
            layout: &bind_group_layout(device),
            entries: &[make_entry(0, &spheres)],
        })
    }
}

pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene"),
        entries: &[make_layout_entry(0)],
    })
}

fn make_buffer<T: NoUninit>(device: &wgpu::Device, data: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(std::any::type_name::<T>()),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn make_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn make_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
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
