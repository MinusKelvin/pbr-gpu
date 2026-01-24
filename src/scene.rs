use std::num::NonZero;
use std::path::Path;

use bytemuck::NoUninit;
use glam::{BVec3, Vec3};
use image::DynamicImage;
use wgpu::util::DeviceExt;

use crate::storage_buffer_entry;

mod light;
mod material;
mod node;
mod shapes;
mod texture;

pub use self::light::*;
pub use self::material::*;
pub use self::node::*;
pub use self::shapes::*;
pub use self::texture::*;

#[derive(Default)]
pub struct Scene {
    pub spheres: Vec<Sphere>,
    pub triangles: Vec<Triangle>,

    pub triangle_vertices: Vec<TriVertex>,

    pub bvh_nodes: Vec<BvhNode>,
    pub transform_nodes: Vec<TransformNode>,
    pub primitive_nodes: Vec<PrimitiveNode>,

    pub constant_float_tex: Vec<ConstantFloatTexture>,
    pub constant_rgb_tex: Vec<ConstantRgbTexture>,
    pub constant_spectrum_tex: Vec<ConstantSpectrumTexture>,
    pub image_rgb_tex: Vec<ImageRgbTexture>,
    pub scale_tex: Vec<ScaleTexture>,
    pub mix_tex: Vec<MixTexture>,
    pub checkerboard_tex: Vec<CheckerboardTexture>,

    pub images: Vec<image::DynamicImage>,

    pub diffuse_mat: Vec<DiffuseMaterial>,
    pub diffuse_transmit_mat: Vec<DiffuseTransmitMaterial>,

    pub infinite_lights: Vec<LightId>,

    pub uniform_lights: Vec<UniformLight>,
    pub image_lights: Vec<ImageLight>,
    pub area_lights: Vec<AreaLight>,

    pub root: Option<NodeId>,
}

impl Scene {
    pub fn make_bind_group_layout(&self, device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene"),
            entries: &[
                storage_buffer_entry(0),
                storage_buffer_entry(1),
                storage_buffer_entry(2),
                storage_buffer_entry(32),
                storage_buffer_entry(33),
                storage_buffer_entry(34),
                storage_buffer_entry(35),
                storage_buffer_entry(64),
                storage_buffer_entry(65),
                storage_buffer_entry(66),
                storage_buffer_entry(67),
                wgpu::BindGroupLayoutEntry {
                    binding: 68,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: Some(
                        NonZero::new(self.images.len() as u32).unwrap_or(NonZero::new(1).unwrap()),
                    ),
                },
                storage_buffer_entry(69),
                storage_buffer_entry(70),
                storage_buffer_entry(71),
                storage_buffer_entry(96),
                storage_buffer_entry(97),
                storage_buffer_entry(128),
                storage_buffer_entry(130),
                storage_buffer_entry(131),
                storage_buffer_entry(132),
            ],
        })
    }

    pub fn make_bind_group(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        let spheres = make_buffer(device, &self.spheres);
        let triangles = make_buffer(device, &self.triangles);

        let triangle_vertices = make_buffer(device, &self.triangle_vertices);

        let bvh = make_buffer(device, &self.bvh_nodes);
        let transform = make_buffer(device, &self.transform_nodes);
        let primitive = make_buffer(device, &self.primitive_nodes);

        let constant_float_tex = make_buffer(device, &self.constant_float_tex);
        let constant_rgb_tex = make_buffer(device, &self.constant_rgb_tex);
        let constant_spectrum_tex = make_buffer(device, &self.constant_spectrum_tex);
        let image_rgb_tex = make_buffer(device, &self.image_rgb_tex);
        let scale_tex = make_buffer(device, &self.scale_tex);
        let mix_tex = make_buffer(device, &self.mix_tex);
        let checkerboard_tex = make_buffer(device, &self.checkerboard_tex);

        let diffuse_mat = make_buffer(device, &self.diffuse_mat);
        let diffuse_transmit_mat = make_buffer(device, &self.diffuse_transmit_mat);

        let infinite_lights = make_buffer(device, &self.infinite_lights);

        let uniform_lights = make_buffer(device, &self.uniform_lights);
        let image_lights = make_buffer(device, &self.image_lights);
        let area_lights = make_buffer(device, &self.area_lights);

        let root = make_buffer(device, &[self.root.unwrap()]);

        let empty = [DynamicImage::new(1, 1, image::ColorType::Rgba8)];
        let images = match self.images.is_empty() {
            true => empty.iter(),
            false => self.images.iter(),
        };

        let views: Vec<_> = images
            .map(|img| {
                let mut desc = wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: img.width(),
                        height: img.height(),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                };

                let mut img_rgba32f;
                let mut img_rgba8;

                let data = if img.as_flat_samples_f32().is_some() {
                    img_rgba32f = img.to_rgba32f();
                    desc.format = wgpu::TextureFormat::Rgba32Float;
                    bytemuck::cast_slice(&img_rgba32f)
                } else {
                    img_rgba8 = img.to_rgba8();
                    bytemuck::cast_slice(&img_rgba8)
                };

                let texture = device.create_texture_with_data(
                    queue,
                    &desc,
                    wgpu::util::TextureDataOrder::LayerMajor,
                    data,
                );

                texture.create_view(&Default::default())
            })
            .collect();
        let views_refs: Vec<_> = views.iter().collect();

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene"),
            layout,
            entries: &[
                make_entry(0, &spheres),
                make_entry(1, &triangles),
                make_entry(2, &triangle_vertices),
                make_entry(32, &root),
                make_entry(33, &bvh),
                make_entry(34, &transform),
                make_entry(35, &primitive),
                make_entry(64, &constant_float_tex),
                make_entry(65, &constant_rgb_tex),
                make_entry(66, &constant_spectrum_tex),
                make_entry(67, &image_rgb_tex),
                wgpu::BindGroupEntry {
                    binding: 68,
                    resource: wgpu::BindingResource::TextureViewArray(&views_refs),
                },
                make_entry(69, &scale_tex),
                make_entry(70, &mix_tex),
                make_entry(71, &checkerboard_tex),
                make_entry(96, &diffuse_mat),
                make_entry(97, &diffuse_transmit_mat),
                make_entry(128, &infinite_lights),
                make_entry(130, &uniform_lights),
                make_entry(131, &image_lights),
                make_entry(132, &area_lights),
            ],
        })
    }

    pub fn add_image(&mut self, path: &Path) -> Option<u32> {
        let Ok(img) = image::open(path)
            .inspect_err(|e| println!("Could not load image {}: {e}", path.display()))
        else {
            return None;
        };
        let id = self.images.len() as u32;
        self.images.push(img);
        Some(id)
    }
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
