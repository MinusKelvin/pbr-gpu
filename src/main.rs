use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use clap::Parser;
use clap::builder::{StringValueParser, TypedValueParser};
use glam::{Mat3, Mat4, Vec3, Vec4, Vec4Swizzles};
use image::{Rgb, RgbImage, Rgba32FImage};
use rand::prelude::SliceRandom;
use rand_pcg::Pcg64;
use wgpu::util::DeviceExt;

mod guided_state;
mod loader;
mod megakernel;
mod options;
mod scene;
mod shader;
mod spectrum;

#[derive(Parser)]
struct Options {
    #[clap(short = 'W', long)]
    width: Option<u32>,
    #[clap(short = 'H', long)]
    height: Option<u32>,

    #[clap(short, long)]
    samples: Option<u32>,
    #[clap(short, long, value_parser = StringValueParser::new().try_map(parse_time))]
    time: Option<Duration>,

    #[clap(long, default_value = "simple")]
    integrator: String,

    #[clap(long, default_value = "independent")]
    sampler: String,

    #[clap(long, default_value = "1")]
    scale: f32,

    #[clap(long, default_value = "0")]
    sample_offset: u32,

    #[clap(long)]
    scene_stats: bool,

    scene: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let options = Options::parse();

    let spectrum_data = spectrum::load_data().unwrap();

    let (mut render_options, scene) = loader::pbrt::load_pbrt_scene(&spectrum_data, &options.scene);

    if let Some(width) = options.width {
        render_options.width = width;
    }
    if let Some(height) = options.height {
        render_options.height = height;
    }
    if let Some(time) = options.time {
        render_options.samples = u32::MAX;
        render_options.time = time;
    }
    if let Some(samples) = options.samples {
        render_options.samples = samples;
    }

    let squish = Mat4::from_scale(Vec3::new(
        (render_options.width as f32 / render_options.height as f32).max(1.0),
        (render_options.height as f32 / render_options.width as f32).max(1.0),
        1.0,
    ));
    render_options.camera.ndc_to_camera.mul_assign(squish);

    if options.scene_stats {
        scene.print_stats();
    }

    let instance = wgpu::Instance::new(&Default::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::SHADER_INT64
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | wgpu::Features::FLOAT32_FILTERABLE
            | wgpu::Features::SHADER_FLOAT32_ATOMIC
            | wgpu::Features::CLEAR_TEXTURE
            | wgpu::Features::IMMEDIATES,
        required_limits: wgpu::Limits {
            max_immediate_size: 64,
            max_storage_buffer_binding_size: (2 << 30) - 4,
            max_buffer_size: (2 << 30) - 4,
            max_storage_buffers_per_shader_stage: 128,
            max_binding_array_elements_per_shader_stage: 4096,
            ..wgpu::Limits::default().using_resolution(adapter.limits())
        },
        ..Default::default()
    }))?;

    let sampler_data = match options.sampler.as_str() {
        "independent" => vec![0; 4],
        "roberts" => bytemuck::pod_collect_to_vec(&roberts_sampler_data()),
        s => unreachable!("invalid sampler `{s}`"),
    };

    let film_desc = wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: render_options.width,
            height: render_options.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };
    let mean = device.create_texture(&film_desc);
    let variance = device.create_texture(&film_desc);

    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&render_options.camera),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let sampler_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &sampler_data,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let rgb_coeff_texture = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: spectrum::RGB_COEFF_N,
                height: spectrum::RGB_COEFF_N,
                depth_or_array_layers: spectrum::RGB_COEFF_N,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(&spectrum_data.rgb_coeffs),
    );

    let linear_clamp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 0.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    });

    let linear_wrap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 0.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    });

    let statics_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            storage_buffer_entry(8),
            storage_buffer_entry(16),
            wgpu::BindGroupLayoutEntry {
                binding: 24,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 25,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    let statics_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &statics_bg_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &mean.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &variance.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: sampler_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 16,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 24,
                resource: wgpu::BindingResource::Sampler(&linear_clamp_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 25,
                resource: wgpu::BindingResource::Sampler(&linear_wrap_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 32,
                resource: wgpu::BindingResource::TextureView(
                    &rgb_coeff_texture.create_view(&Default::default()),
                ),
            },
        ],
    });

    let start = Instant::now();

    let num_samples = megakernel::run(
        &options,
        &device,
        &queue,
        &scene,
        render_options,
        &statics_bg_layout,
        &statics_bg,
        &mean,
        &variance,
    )?;

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let took = start.elapsed();

    if std::env::var_os("MESA_VK_TRACE_PER_SUBMIT").is_some() {
        std::thread::sleep(Duration::from_secs(1));
    }

    let stats = collect_stats(&device, &queue, &mean, &variance, took);

    println!(
        "Took {:.2} seconds ({:.3?} / sample)",
        took.as_secs_f64(),
        took / num_samples,
    );
    println!("Average variance: {}", stats.avg_variance);
    println!("Average relative variance: {}", stats.avg_rel_variance);
    println!("Average relative error: {}", stats.avg_rel_error.sqrt());
    println!("Efficiency: {}", stats.efficiency);

    xyz_to_srgb(&stats.mean_image, options.scale)
        .save("img.png")
        .unwrap();

    Ok(())
}

fn download_texture(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    downloaded: impl FnOnce(Vec<Vec4>) + Send + 'static,
) {
    let bytes_per_row = (texture.width() * 16).next_multiple_of(256);

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes_per_row as u64 * texture.height() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        texture.size(),
    );

    let buf = buffer.clone();
    let width = texture.width() as usize;
    encoder.map_buffer_on_submit(&buf, wgpu::MapMode::Read, .., move |result| {
        result.unwrap();

        let data = buffer.get_mapped_range(..);
        let data: &[Vec4] = bytemuck::cast_slice(&data);
        let data: Vec<_> = data
            .chunks_exact(bytes_per_row as usize / 16)
            .flat_map(|chunk| chunk[..width].iter().copied())
            .collect();

        downloaded(data);
    });
}

fn download_buffer(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    buffer: &wgpu::Buffer,
    downloaded: impl FnOnce(&[u8]) + Send + 'static,
) {
    let dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&buffer, 0, &dst_buffer, 0, buffer.size());

    let buffer = dst_buffer.clone();
    encoder.map_buffer_on_submit(&dst_buffer, wgpu::MapMode::Read, .., move |result| {
        result.unwrap();
        downloaded(&buffer.get_mapped_range(..));
    });
}

fn storage_buffer_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn writable_storage_buffer_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[derive(Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C)]
struct ProjectiveCamera {
    ndc_to_camera: Transform,
    world_to_camera: Transform,
    lens_radius: f32,
    focal_distance: f32,
    orthographic: u32,
    _padding: u32,
}

#[derive(Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C)]
struct Transform {
    m: Mat4,
    m_inv: Mat4,
}

impl Transform {
    fn from_mat4(value: Mat4) -> Self {
        Self {
            m: value,
            m_inv: value.inverse(),
        }
    }

    fn from_mat4_inverse(inverse: Mat4) -> Self {
        Self {
            m: inverse.inverse(),
            m_inv: inverse,
        }
    }

    fn mul_assign(&mut self, other: Mat4) {
        self.m = self.m * other;
        self.m_inv = other.inverse() * self.m_inv;
    }
}

trait ExtraState {
    fn add_bind_group_layouts<'a>(&'a mut self, bg_layouts: &mut Vec<&'a wgpu::BindGroupLayout>);
    fn setup_pass(&mut self, pass: &mut wgpu::ComputePass);
    fn before_sample(
        &mut self,
        sample: u32,
        time: Duration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mean: &wgpu::Texture,
        variance: &wgpu::Texture,
    );
}

impl ExtraState for () {
    fn add_bind_group_layouts<'a>(&'a mut self, _bg_layouts: &mut Vec<&'a wgpu::BindGroupLayout>) {}
    fn setup_pass(&mut self, _pass: &mut wgpu::ComputePass) {}
    fn before_sample(
        &mut self,
        _sample: u32,
        _time: Duration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _mean: &wgpu::Texture,
        _variance: &wgpu::Texture,
    ) {
    }
}

fn parse_time(mut s: String) -> Result<Duration, std::num::ParseFloatError> {
    s.make_ascii_lowercase();
    let number = s.trim_end_matches(char::is_alphabetic);
    let suffix = s[number.len()..].trim();
    let number = number.parse::<f64>()?;

    let unit_seconds = match suffix {
        "ms" => 0.001,
        "" | "s" | "sec" | "second" | "seconds" => 1.0,
        "min" | "mins" | "minute" | "minutes" => 60.0,
        "h" | "hr" | "hrs" | "hour" | "hours" => 3600.0,
        "d" | "day" | "days" => 24.0 * 3600.0,
        _ => 1.0,
    };

    Ok(Duration::from_secs_f64(number * unit_seconds))
}

struct ImageStats {
    mean_image: Rgba32FImage,
    avg_variance: f64,
    avg_rel_variance: f64,
    avg_rel_error: f64,
    efficiency: f64,
}

fn collect_stats(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mean: &wgpu::Texture,
    variance: &wgpu::Texture,
    time: Duration,
) -> ImageStats {
    let width = mean.width();
    let height = mean.height();

    let mut encoder = device.create_command_encoder(&Default::default());

    let downloaded = Arc::new(Mutex::new((vec![], vec![])));

    let dl = downloaded.clone();
    download_texture(&device, &mut encoder, &mean, move |data| {
        dl.lock().unwrap().0 = data;
    });

    let dl = downloaded.clone();
    download_texture(&device, &mut encoder, &variance, move |data| {
        dl.lock().unwrap().1 = data;
    });

    queue.submit([encoder.finish()]);

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let (mean, variance) = Arc::into_inner(downloaded).unwrap().into_inner().unwrap();

    let mut avg_variance = 0.0;
    let mut avg_rel_variance = 0.0;
    let mut avg_rel_error = 0.0;
    let mut avg_spp = 0.0;
    for (&mean, &s) in mean.iter().zip(&variance) {
        let samples = mean.w;
        let mean = mean.xyz();
        let s = s.xyz();

        let var = if samples == 1.0 {
            Vec3::INFINITY
        } else {
            s / (samples - 1.0)
        };

        let rel_var = var / mean;
        let rel_var = Vec3::select(rel_var.is_finite_mask(), rel_var, Vec3::ZERO);
        let rel_err = rel_var / samples;

        avg_variance += var.element_sum() as f64 / 3.0;
        avg_rel_variance += rel_var.element_sum() as f64 / 3.0;
        avg_rel_error += rel_err.element_sum() as f64 / 3.0;
        avg_spp += samples as f64;
    }
    let avg_variance = avg_variance / mean.len() as f64;
    let avg_rel_variance = avg_rel_variance / mean.len() as f64;
    let avg_rel_error = avg_rel_error / mean.len() as f64;
    let avg_spp = avg_spp / mean.len() as f64;

    let avg_sample_time = time.as_secs_f64() / avg_spp;

    let efficiency = 1.0 / (avg_rel_variance * avg_sample_time);

    let mut invalid_pixel = None;

    let mean_image = Rgba32FImage::from_vec(
        width,
        height,
        mean.into_iter()
            .enumerate()
            .inspect(|&(i, raw)| {
                if !raw.is_finite() {
                    invalid_pixel = Some(i);
                }
            })
            .flat_map(|(_, v)| v.to_array())
            .collect(),
    )
    .unwrap();

    if let Some(i) = invalid_pixel {
        println!("Warning: Pixel {i} had non-finite value");
    }

    ImageStats {
        mean_image,
        avg_variance,
        avg_rel_variance,
        avg_rel_error,
        efficiency,
    }
}

fn xyz_to_srgb(xyz: &Rgba32FImage, scale: f32) -> RgbImage {
    const SRGB_TO_XYZ_T: Mat3 = Mat3::from_cols_array_2d(&[
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ]);
    let xyz_to_srgb = SRGB_TO_XYZ_T.transpose().inverse();

    RgbImage::from_fn(xyz.width(), xyz.height(), |x, y| {
        let rgb = xyz_to_srgb * Vec4::from_array(xyz.get_pixel(x, y).0).xyz() * scale;
        if !rgb.is_finite() {
            return Rgb([255, 0, 255]);
        }
        let low = rgb * 12.92;
        let high = rgb.powf(1.0 / 2.4) * 1.055 - 0.055;
        let srgb = Vec3::select(rgb.cmplt(Vec3::splat(0.0031308)), low, high);
        Rgb((srgb * 255.0).as_u8vec3().to_array())
    })
}

fn roberts_sampler_data() -> Vec<u32> {
    const DIM: usize = 256;

    let inv_d1 = 1.0 / (DIM as f64 + 1.0);

    let mut phi = 2.0f64;
    for _ in 0..25 {
        phi = (1.0 + phi).powf(inv_d1);
    }

    let mut alphas: Vec<_> = (0..DIM)
        .map(|i| {
            let alpha = 1.0 / phi.powi((i + 1) as i32);
            let alpha = 1.0 - alpha;
            (alpha * 32f64.exp2()).round() as u32
        })
        .collect();

    alphas.shuffle(&mut Pcg64::new(
        0xcafef00dd15ea5e5,
        0xa02bdbf7bb3c0a7ac28fa16a64abf96,
    ));

    alphas
}
