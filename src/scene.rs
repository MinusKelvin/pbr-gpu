use std::collections::HashMap;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;
use std::num::NonZero;
use std::ops::Range;
use std::path::Path;

use bytemuck::NoUninit;
use glam::Vec4;
use glam::{BVec3, Vec3};
use image::DynamicImage;
use image::ImageBuffer;
use image::Luma;
use image::Pixel;
use image::Rgb32FImage;
use image::Rgba32FImage;
use image::RgbaImage;
use wgpu::util::DeviceExt;

use crate::Transform;
use crate::spectrum::SpectrumData;
use crate::storage_buffer_entry;

mod light;
mod light_sampler;
mod material;
mod other;
mod shapes;
mod spectra;
mod texture;

pub use self::light::*;
pub use self::light_sampler::*;
pub use self::material::*;
pub use self::other::*;
pub use self::shapes::*;
pub use self::spectra::*;
pub use self::texture::*;

type Luma32FImage = ImageBuffer<Luma<f32>, Vec<f32>>;

#[derive(Default)]
pub struct Scene {
    // pub spheres: Vec<Sphere>,
    pub triangles: Vec<Triangle>,

    pub triangle_vertices: Vec<TriVertex>,

    pub triangle_properties: Vec<TriangleProperties>,

    pub objects: Vec<Vec<Range<usize>>>,
    pub instances: Vec<(usize, Transform)>,

    pub images: Vec<ImageData>,

    pub diffuse_mat: Vec<DiffuseMaterial>,
    pub diffuse_transmit_mat: Vec<DiffuseTransmitMaterial>,
    pub conductor_mat: Vec<ConductorMaterial>,
    pub dielectric_mat: Vec<DielectricMaterial>,
    pub thin_dielectric_mat: Vec<ThinDielectricMaterial>,
    pub metallic_workflow_mat: Vec<MetallicWorkflowMaterial>,
    pub mix_mat: Vec<MixMaterial>,

    pub infinite_lights: Vec<LightId>,

    pub uniform_lights: Vec<UniformLight>,
    pub image_lights: Vec<ImageLight>,
    pub area_lights: Vec<AreaLight>,
    pub distant_lights: Vec<DistantLight>,

    pub table_spectra: Vec<TableSpectrum>,
    pub constant_spectra: Vec<ConstantSpectrum>,
    pub rgb_albedo_spectra: Vec<RgbAlbedoSpectrum>,
    pub rgb_illuminant_spectra: Vec<RgbIlluminantSpectrum>,
    pub blackbody_spectra: Vec<BlackbodySpectrum>,
    pub piecewise_linear_spectra: Vec<PiecewiseLinearSpectrum>,

    pub float_data: Vec<f32>,

    pub uniform_light_samplers: Vec<UniformLightSampler>,
    pub uniform_light_sampler_data: Vec<LightId>,
    pub power_light_samplers: Vec<PowerLightSampler>,
    pub power_light_sampler_data: Vec<PlsAliasBucket>,

    pub root_ls: Option<LightSamplerId>,

    pub named_spectra: HashMap<&'static str, SpectrumId>,

    float_texture_match: String,
    spectrum_texture_match: String,
    float_code: HashMap<String, u32>,
    spectrum_code: HashMap<String, u32>,
}

pub enum ImageData {
    Float(Luma32FImage),
    FloatRgb(Rgba32FImage),
    Srgb(RgbaImage),
    UnormRgb(RgbaImage),
}

impl Scene {
    pub fn new(builtin: &SpectrumData) -> Self {
        let mut this = Scene::default();
        // empty slot
        this.infinite_lights.push(LightId::ZERO);
        this.add_table_spectrum(*builtin.cie_x);
        this.add_table_spectrum(*builtin.cie_y);
        this.add_table_spectrum(*builtin.cie_z);
        let v = this.add_table_spectrum(*builtin.d65);
        this.named_spectra.insert("stdillum-D65", v);
        this.add_table_spectrum(TableSpectrum { data: [1.0; 471] });
        for (name, data) in &builtin.iors {
            let v = this.add_piecewise_linear_spectrum(data);
            this.named_spectra.insert(name, v);
        }
        this
    }

    #[rustfmt::skip]
    pub fn print_stats(&self) {
        println!("Shapes");
        // println!("  Spheres           {}", human_size_of(&self.spheres));
        println!("  Triangles         {}", human_size_of(&self.triangles));
        println!("  Tri verts         {}", human_size_of(&self.triangle_vertices));
        println!("  Tri properties    {}", human_size_of(&self.triangle_properties));
        println!("Texture Metadata");
        println!("  Spectrum          {}", self.spectrum_code.len());
        println!("  Float             {}", self.float_code.len());
        println!("  Image data        {}", human_size(self.images.iter().map(|img| match img {
            ImageData::Float(img) => std::mem::size_of_val(img.as_raw().as_slice()),
            ImageData::FloatRgb(img) => std::mem::size_of_val(img.as_raw().as_slice()),
            ImageData::UnormRgb(img) => std::mem::size_of_val(img.as_raw().as_slice()),
            ImageData::Srgb(img) => std::mem::size_of_val(img.as_raw().as_slice()),
        }).sum()));
        println!("Materials");
        println!("  Diffuse           {}", human_size_of(&self.diffuse_mat));
        println!("  Diffuse Transmit  {}", human_size_of(&self.diffuse_transmit_mat));
        println!("  Conductor         {}", human_size_of(&self.conductor_mat));
        println!("  Dielectric        {}", human_size_of(&self.dielectric_mat));
        println!("  Thin Dielectric   {}", human_size_of(&self.thin_dielectric_mat));
        println!("  Metallic Workflow {}", human_size_of(&self.metallic_workflow_mat));
        println!("  Mix               {}", human_size_of(&self.mix_mat));
        println!("Lights");
        println!("  Inf Uniform       {}", human_size_of(&self.uniform_lights));
        println!("  Inf Image         {}", human_size_of(&self.image_lights));
        println!("  Area              {}", human_size_of(&self.area_lights));
        println!("  Distant           {}", human_size_of(&self.distant_lights));
        println!("  Inf Light List    {}", human_size_of(&self.infinite_lights));
        println!("Light Samplers");
        println!("  Uniform           {}", human_size_of(&self.uniform_light_samplers));
        println!("  Uniform Data      {}", human_size_of(&self.uniform_light_sampler_data));
        println!("  Power             {}", human_size_of(&self.power_light_samplers));
        println!("  Power Data        {}", human_size_of(&self.power_light_sampler_data));
        println!("Spectra");
        println!("  Table             {}", human_size_of(&self.table_spectra));
        println!("  Constant          {}", human_size_of(&self.constant_spectra));
        println!("  Rgb               {}", human_size_of(&self.rgb_albedo_spectra));
        println!("  Rgb Illuminant    {}", human_size_of(&self.rgb_illuminant_spectra));
        println!("  Blackbody         {}", human_size_of(&self.blackbody_spectra));
        println!("  Piecewise Linear  {}", human_size_of(&self.piecewise_linear_spectra));
        println!("Misc Data           {}", human_size_of(&self.float_data));
    }

    pub fn make_bind_group_layout(&self, device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene"),
            entries: &[
                // storage_buffer_entry(0),
                storage_buffer_entry(1),
                storage_buffer_entry(2),
                storage_buffer_entry(3),
                storage_buffer_entry(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 68,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: Some(
                        NonZero::new(self.images.len() as u32).unwrap_or(NonZero::new(1).unwrap()),
                    ),
                },
                storage_buffer_entry(96),
                storage_buffer_entry(97),
                storage_buffer_entry(98),
                storage_buffer_entry(99),
                storage_buffer_entry(100),
                storage_buffer_entry(101),
                storage_buffer_entry(102),
                storage_buffer_entry(128),
                storage_buffer_entry(129),
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
                storage_buffer_entry(224),
                storage_buffer_entry(225),
                storage_buffer_entry(226),
                storage_buffer_entry(227),
                storage_buffer_entry(228),
            ],
        })
    }

    pub fn make_bind_group(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        // let spheres = make_buffer(device, &self.spheres);
        let triangles = make_buffer_blas(device, &self.triangles);

        let triangle_vertices = make_buffer_blas(device, &self.triangle_vertices);
        let triangle_properties = make_buffer(device, &self.triangle_properties);

        let diffuse_mat = make_buffer(device, &self.diffuse_mat);
        let diffuse_transmit_mat = make_buffer(device, &self.diffuse_transmit_mat);
        let conductor_mat = make_buffer(device, &self.conductor_mat);
        let dielectric_mat = make_buffer(device, &self.dielectric_mat);
        let thin_dielectric_mat = make_buffer(device, &self.thin_dielectric_mat);
        let metallic_workflow_mat = make_buffer(device, &self.metallic_workflow_mat);
        let mix_mat = make_buffer(device, &self.mix_mat);

        let infinite_lights = make_buffer(device, &self.infinite_lights);

        let uniform_lights = make_buffer(device, &self.uniform_lights);
        let image_lights = make_buffer(device, &self.image_lights);
        let area_lights = make_buffer(device, &self.area_lights);
        let distant_lights = make_buffer(device, &self.distant_lights);

        let table_spectra = make_buffer(device, &self.table_spectra);
        let constant_spectra = make_buffer(device, &self.constant_spectra);
        let rgb_albedo_spectra = make_buffer(device, &self.rgb_albedo_spectra);
        let rgb_illuminant_spectra = make_buffer(device, &self.rgb_illuminant_spectra);
        let blackbody_spectra = make_buffer(device, &self.blackbody_spectra);
        let piecewise_linear_spectra = make_buffer(device, &self.piecewise_linear_spectra);

        let float_data = make_buffer(device, &self.float_data);

        let uniform_light_samplers = make_buffer(device, &self.uniform_light_samplers);
        let uniform_light_sampler_data = make_buffer(device, &self.uniform_light_sampler_data);
        let power_light_samplers = make_buffer(device, &self.power_light_samplers);
        let power_light_sampler_data = make_buffer(device, &self.power_light_sampler_data);

        let root_ls = make_buffer(device, &[self.root_ls.unwrap()]);

        let empty = [ImageData::Srgb(RgbaImage::new(1, 1))];
        let images = match self.images.is_empty() {
            true => empty.iter(),
            false => self.images.iter(),
        };

        let mut blases = vec![];
        let mut obj_sizes = vec![];
        let mut triangle_offsets = vec![];
        let mut blas_triangle_offset_index = vec![];
        for obj in &self.objects {
            let tri_offset_index = triangle_offsets.len();
            for range in obj {
                triangle_offsets.push(range.start as u32);
            }
            let sizes = obj
                .iter()
                .map(|range| wgpu::BlasTriangleGeometrySizeDescriptor {
                    vertex_format: wgpu::VertexFormat::Float32x3,
                    vertex_count: self.triangle_vertices.len() as u32,
                    index_format: Some(wgpu::IndexFormat::Uint32),
                    index_count: Some(3 * (range.end - range.start) as u32),
                    flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                })
                .collect::<Vec<_>>();
            let blas = device.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: None,
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: sizes.clone(),
                },
            );
            blases.push(blas);
            obj_sizes.push(sizes);
            blas_triangle_offset_index.push(tri_offset_index as u32);
        }

        let mut blas_builds = vec![];
        for ((blas, obj), sizes) in blases.iter().zip(&self.objects).zip(&obj_sizes) {
            let geometry = obj
                .iter()
                .zip(sizes)
                .map(|(range, size)| wgpu::BlasTriangleGeometry {
                    size,
                    vertex_buffer: &triangle_vertices,
                    first_vertex: 0,
                    vertex_stride: std::mem::size_of::<TriVertex>() as u64,
                    index_buffer: Some(&triangles),
                    first_index: Some(range.start as u32 * 3),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                })
                .collect();
            blas_builds.push(wgpu::BlasBuildEntry {
                blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(geometry),
            });
        }

        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: None,
            max_instances: self.instances.len() as u32,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        for (tlas_instance, &(obj_id, tform)) in tlas[0..self.instances.len()]
            .iter_mut()
            .zip(&self.instances)
        {
            assert_eq!(tform.m_inv.row(3), Vec4::W);
            let tform = tform.m_inv.transpose().to_cols_array()[..12]
                .try_into()
                .unwrap();
            *tlas_instance = Some(wgpu::TlasInstance::new(
                &blases[obj_id],
                tform,
                blas_triangle_offset_index[obj_id],
                !0,
            ));
        }

        let triangle_offsets = make_buffer(device, &triangle_offsets);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.build_acceleration_structures(blas_builds.iter(), [&tlas]);
        queue.submit([encoder.finish()]);

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
                    ImageData::UnormRgb(img) => (
                        img.width(),
                        img.height(),
                        wgpu::TextureFormat::Rgba8Unorm,
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
                // make_entry(0, &spheres),
                make_entry(1, &triangles),
                make_entry(2, &triangle_vertices),
                make_entry(3, &triangle_properties),
                make_entry(4, &triangle_offsets),
                wgpu::BindGroupEntry {
                    binding: 32,
                    resource: tlas.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 68,
                    resource: wgpu::BindingResource::TextureViewArray(&views_refs),
                },
                make_entry(96, &diffuse_mat),
                make_entry(97, &diffuse_transmit_mat),
                make_entry(98, &conductor_mat),
                make_entry(99, &dielectric_mat),
                make_entry(100, &thin_dielectric_mat),
                make_entry(101, &metallic_workflow_mat),
                make_entry(102, &mix_mat),
                make_entry(128, &infinite_lights),
                make_entry(129, &uniform_lights),
                make_entry(130, &image_lights),
                make_entry(131, &area_lights),
                make_entry(132, &distant_lights),
                make_entry(160, &table_spectra),
                make_entry(161, &constant_spectra),
                make_entry(162, &rgb_albedo_spectra),
                make_entry(163, &rgb_illuminant_spectra),
                make_entry(164, &blackbody_spectra),
                make_entry(165, &piecewise_linear_spectra),
                make_entry(192, &float_data),
                make_entry(224, &root_ls),
                make_entry(225, &uniform_light_samplers),
                make_entry(226, &uniform_light_sampler_data),
                make_entry(227, &power_light_samplers),
                make_entry(228, &power_light_sampler_data),
            ],
        })
    }

    pub fn generated_texture_shader_code(&self) -> String {
        let spectrum_cases = &self.spectrum_texture_match;
        let float_cases = &self.float_texture_match;
        format!(
            "
fn spectrum_texture_evaluate(id: SpectrumTextureId, uv: vec2f, wavelengths: Wavelengths) -> vec4f {{
    switch id.id {{
        {spectrum_cases}
        default {{ return vec4f(); }}
    }}
}}
fn float_texture_evaluate(id: FloatTextureId, uv: vec2f) -> f32 {{
    switch id.id {{
        {float_cases}
        default {{ return 0; }}
    }}
}}"
        )
    }

    pub fn add_image(&mut self, path: &Path, float: bool, no_gamma: bool) -> Option<u32> {
        let img = match path.extension().and_then(|s| s.to_str()) {
            Some("pfm") => load_pfm_image(path),
            _ => image::open(path),
        };
        let Ok(img) = img.inspect_err(|e| println!("Could not load image {}: {e}", path.display()))
        else {
            return None;
        };
        let id = self.images.len() as u32;
        self.images.push(match img {
            _ if float && img.has_alpha() => {
                let data = img.to_luma_alpha32f();
                let data = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
                    Luma([data.get_pixel(x, y).alpha()])
                });
                ImageData::Float(data)
            }
            DynamicImage::ImageLuma16(_) | DynamicImage::ImageLuma8(_) if float => {
                ImageData::Float(img.to_luma32f())
            }
            _ if float => {
                println!(
                    "creating float texture from color image without alpha is suspect ({})",
                    path.display()
                );
                let data = img.to_rgba32f();
                let data = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
                    Luma([data.get_pixel(x, y).0[0]])
                });
                ImageData::Float(data)
            }
            _ if img.as_flat_samples_f32().is_some() => ImageData::FloatRgb(img.to_rgba32f()),
            _ if no_gamma => ImageData::UnormRgb(img.to_rgba8()),
            _ => ImageData::Srgb(img.to_rgba8()),
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
            ImageData::UnormRgb(img) => (
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

fn make_buffer_blas<T: NoUninit>(device: &wgpu::Device, data: &[T]) -> wgpu::Buffer {
    let empty = vec![0; std::mem::size_of::<T>()];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(std::any::type_name::<T>()),
        contents: match data.is_empty() {
            true => &empty,
            false => bytemuck::cast_slice(data),
        },
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
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

fn load_pfm_image(path: &Path) -> image::ImageResult<DynamicImage> {
    use image::error::*;

    let fmt_hint = ImageFormatHint::Name("PFM".to_string());

    let mut buf_reader = BufReader::new(File::open(path)?);
    let mut buf = String::new();

    buf_reader.read_line(&mut buf)?;
    let is_rgb = match buf.trim() {
        "PF" => true,
        "Pf" => false,
        _ => {
            return Err(ImageError::Decoding(DecodingError::new(
                fmt_hint,
                "invalid pfm type",
            )));
        }
    };

    buf.clear();
    buf_reader.read_line(&mut buf)?;
    let (width, height) =
        buf.trim()
            .split_once(' ')
            .ok_or(ImageError::Decoding(DecodingError::new(
                fmt_hint.clone(),
                "expected width and height to be specified",
            )))?;
    let width = width
        .parse()
        .map_err(|e| ImageError::Decoding(DecodingError::new(fmt_hint.clone(), e)))?;
    let height = height
        .parse()
        .map_err(|e| ImageError::Decoding(DecodingError::new(fmt_hint.clone(), e)))?;

    buf.clear();
    buf_reader.read_line(&mut buf)?;
    let wack = buf
        .trim()
        .parse::<f32>()
        .map_err(|e| ImageError::Decoding(DecodingError::new(fmt_hint.clone(), e)))?;
    let is_le = match () {
        _ if wack.is_sign_positive() => false,
        _ if wack.is_sign_negative() => true,
        _ => {
            return Err(ImageError::Decoding(DecodingError::new(
                fmt_hint,
                "invalid byte order specifier",
            )));
        }
    };

    let mut data = vec![
        0.0;
        width as usize
            * height as usize
            * match is_rgb {
                true => 3,
                false => 1,
            }
    ];

    buf_reader.read_exact(bytemuck::cast_slice_mut(&mut data))?;

    if !is_le {
        for v in bytemuck::cast_slice_mut::<_, u32>(&mut data) {
            *v = v.swap_bytes();
        }
    }

    Ok(match is_rgb {
        true => Rgb32FImage::from_vec(width, height, data).unwrap().into(),
        false => Luma32FImage::from_vec(width, height, data).unwrap().into(),
    })
}

fn human_size_of<T>(data: &[T]) -> String {
    human_size(std::mem::size_of_val(data))
}

fn human_size(size: usize) -> String {
    let size = size as f64;
    let kib = size / 1024.0;
    let mib = kib / 1024.0;
    let gib = mib / 1024.0;
    if gib > 1.0 {
        format!("{gib:7.1} GiB")
    } else if mib > 1.0 {
        format!("{mib:7.1} MiB")
    } else if kib > 1.0 {
        format!("{kib:7.1} KiB")
    } else {
        format!("{size:7.1} B")
    }
}
