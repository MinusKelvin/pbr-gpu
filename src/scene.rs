use std::collections::HashMap;
use std::num::NonZero;
use std::path::Path;

use bytemuck::NoUninit;
use glam::{BVec3, Vec3};
use image::DynamicImage;
use image::GenericImageView;
use image::ImageBuffer;
use image::Luma;
use image::Pixel;
use image::Rgba32FImage;
use image::RgbaImage;
use wgpu::util::DeviceExt;

use crate::spectrum::SpectrumData;
use crate::storage_buffer_entry;

mod light;
mod material;
mod node;
mod other;
mod shapes;
mod spectra;
mod texture;

pub use self::light::*;
pub use self::material::*;
pub use self::node::*;
pub use self::other::*;
pub use self::shapes::*;
pub use self::spectra::*;
pub use self::texture::*;

#[derive(Default)]
pub struct Scene {
    pub spheres: Vec<Sphere>,
    pub triangles: Vec<Triangle>,

    pub triangle_vertices: Vec<TriVertex>,

    pub bvh_nodes: Vec<BvhNode>,
    pub transform_nodes: Vec<TransformNode>,
    pub primitive_nodes: Vec<PrimitiveNode>,

    pub constant_tex: Vec<ConstantTexture>,
    pub image_float_tex: Vec<ImageFloatTexture>,
    pub image_rgb_tex: Vec<ImageRgbTexture>,
    pub scale_tex: Vec<ScaleTexture>,
    pub mix_tex: Vec<MixTexture>,
    pub checkerboard_tex: Vec<CheckerboardTexture>,

    pub images: Vec<ImageData>,

    pub diffuse_mat: Vec<DiffuseMaterial>,
    pub diffuse_transmit_mat: Vec<DiffuseTransmitMaterial>,
    pub conductor_mat: Vec<ConductorMaterial>,
    pub dielectric_mat: Vec<DielectricMaterial>,

    pub infinite_lights: Vec<LightId>,

    pub uniform_lights: Vec<UniformLight>,
    pub image_lights: Vec<ImageLight>,
    pub area_lights: Vec<AreaLight>,

    pub table_spectra: Vec<TableSpectrum>,
    pub constant_spectra: Vec<ConstantSpectrum>,
    pub rgb_albedo_spectra: Vec<RgbAlbedoSpectrum>,
    pub rgb_illuminant_spectra: Vec<RgbIlluminantSpectrum>,
    pub blackbody_spectra: Vec<BlackbodySpectrum>,
    pub piecewise_linear_spectra: Vec<PiecewiseLinearSpectrum>,

    pub float_data: Vec<f32>,

    pub root: Option<NodeId>,

    pub named_spectra: HashMap<&'static str, SpectrumId>,
}

pub enum ImageData {
    Float(ImageBuffer<Luma<f32>, Vec<f32>>),
    FloatRgb(Rgba32FImage),
    Srgb(RgbaImage),
}

impl Scene {
    pub fn new(builtin: &SpectrumData) -> Self {
        let mut this = Scene::default();
        this.add_table_spectrum(*builtin.cie_x);
        this.add_table_spectrum(*builtin.cie_y);
        this.add_table_spectrum(*builtin.cie_z);
        let v = this.add_table_spectrum(*builtin.d65);
        this.named_spectra.insert("stdillum-D65", v);
        for (name, data) in &builtin.iors {
            let v = this.add_piecewise_linear_spectrum(data);
            this.named_spectra.insert(name, v);
        }
        this
    }

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
                storage_buffer_entry(98),
                storage_buffer_entry(99),
                storage_buffer_entry(128),
                storage_buffer_entry(130),
                storage_buffer_entry(131),
                storage_buffer_entry(132),
                storage_buffer_entry(160),
                storage_buffer_entry(161),
                storage_buffer_entry(162),
                storage_buffer_entry(163),
                storage_buffer_entry(164),
                storage_buffer_entry(165),
                storage_buffer_entry(192),
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

        let constant_tex = make_buffer(device, &self.constant_tex);
        let image_float_tex = make_buffer(device, &self.image_float_tex);
        let image_rgb_tex = make_buffer(device, &self.image_rgb_tex);
        let scale_tex = make_buffer(device, &self.scale_tex);
        let mix_tex = make_buffer(device, &self.mix_tex);
        let checkerboard_tex = make_buffer(device, &self.checkerboard_tex);

        let diffuse_mat = make_buffer(device, &self.diffuse_mat);
        let diffuse_transmit_mat = make_buffer(device, &self.diffuse_transmit_mat);
        let conductor_mat = make_buffer(device, &self.conductor_mat);
        let dielectric_mat = make_buffer(device, &self.dielectric_mat);

        let infinite_lights = make_buffer(device, &self.infinite_lights);

        let uniform_lights = make_buffer(device, &self.uniform_lights);
        let image_lights = make_buffer(device, &self.image_lights);
        let area_lights = make_buffer(device, &self.area_lights);

        let table_spectra = make_buffer(device, &self.table_spectra);
        let constant_spectra = make_buffer(device, &self.constant_spectra);
        let rgb_albedo_spectra = make_buffer(device, &self.rgb_albedo_spectra);
        let rgb_illuminant_spectra = make_buffer(device, &self.rgb_illuminant_spectra);
        let blackbody_spectra = make_buffer(device, &self.blackbody_spectra);
        let piecewise_linear_spectra = make_buffer(device, &self.piecewise_linear_spectra);

        let float_data = make_buffer(device, &self.float_data);

        let root = make_buffer(device, &[self.root.unwrap()]);

        let empty = [ImageData::Srgb(RgbaImage::new(1, 1))];
        let images = match self.images.is_empty() {
            true => empty.iter(),
            false => self.images.iter(),
        };

        let views: Vec<_> = images
            .map(|img| {
                let (width, height, format, data) = match img {
                    ImageData::Float(img) => (
                        img.width(),
                        img.height(),
                        wgpu::TextureFormat::R32Float,
                        bytemuck::cast_slice(&img),
                    ),
                    ImageData::FloatRgb(img) => (
                        img.width(),
                        img.height(),
                        wgpu::TextureFormat::Rgba32Float,
                        bytemuck::cast_slice(&img),
                    ),
                    ImageData::Srgb(img) => (
                        img.width(),
                        img.height(),
                        wgpu::TextureFormat::Rgba8UnormSrgb,
                        bytemuck::cast_slice(&img),
                    ),
                };

                let texture = device.create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: None,
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
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
                make_entry(64, &constant_tex),
                make_entry(66, &image_float_tex),
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
                make_entry(98, &conductor_mat),
                make_entry(99, &dielectric_mat),
                make_entry(128, &infinite_lights),
                make_entry(130, &uniform_lights),
                make_entry(131, &image_lights),
                make_entry(132, &area_lights),
                make_entry(160, &table_spectra),
                make_entry(161, &constant_spectra),
                make_entry(162, &rgb_albedo_spectra),
                make_entry(163, &rgb_illuminant_spectra),
                make_entry(164, &blackbody_spectra),
                make_entry(165, &piecewise_linear_spectra),
                make_entry(192, &float_data),
            ],
        })
    }

    pub fn add_image(&mut self, path: &Path, float: bool) -> Option<u32> {
        let Ok(img) = image::open(path)
            .inspect_err(|e| println!("Could not load image {}: {e}", path.display()))
        else {
            return None;
        };
        let id = self.images.len() as u32;
        self.images.push(match img.as_flat_samples_f32().is_some() {
            _ if float && img.has_alpha() => {
                let data = img.to_luma_alpha32f();
                let data = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
                    Luma([data.get_pixel(x, y).alpha()])
                });
                ImageData::Float(data)
            }
            _ if float => ImageData::Float(img.to_luma32f()),
            true => ImageData::FloatRgb(img.to_rgba32f()),
            false => ImageData::Srgb(img.to_rgba8()),
        });
        Some(id)
    }

    pub fn image_sampling_distribution(&mut self, image: u32) -> TableSampler2d {
        let (width, height, f) = match &self.images[image as usize] {
            ImageData::Float(img) => (img.width(), img.height(), img.to_vec()),
            ImageData::FloatRgb(img) => (
                img.width(),
                img.height(),
                img.pixels().map(|c| c.to_luma().0[0]).collect::<Vec<_>>(),
            ),
            ImageData::Srgb(img) => (
                img.width(),
                img.height(),
                img.pixels().map(|c| c.to_luma().0[0] as f32).collect(),
            ),
        };

        self.add_2d_table_sampler(0.0, 1.0, 0.0, 1.0, width, height, &f)
    }

    pub fn add_float_data(&mut self, data: &[f32]) -> u32 {
        let base = self.float_data.len() as u32;
        self.float_data.extend_from_slice(&data);
        base
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
