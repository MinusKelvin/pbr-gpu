use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use bytemuck::{AnyBitPattern, NoUninit, Pod, Zeroable};
use clap::Parser;
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};
use image::{Rgb, RgbImage};
use ordered_float::OrderedFloat;
use wgpu::PollType;
use wgpu::util::DeviceExt;

use crate::scene::Scene;

mod loader;
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

    #[clap(long, default_value = "simple")]
    integrator: String,

    #[clap(long, default_value = "1")]
    scale: f32,

    #[clap(long, default_value = "0")]
    sample_offset: u32,

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
    if let Some(samples) = options.samples {
        render_options.samples = samples;
    }

    scene.print_stats();

    let instance = wgpu::Instance::new(&Default::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::SHADER_INT64
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

    let mut extra_state = match options.integrator.as_str() {
        "guided" => Box::new(GuidedState::new(&device, &scene, render_options.samples))
            as Box<dyn ExtraState>,
        _ => Box::new(()),
    };

    let flags = [
        ("sampler".to_owned(), "independent".to_owned()),
        ("camera".to_owned(), "projective".to_owned()),
        ("integrator".to_owned(), options.integrator),
    ]
    .into_iter()
    .collect();
    let shader = shader::load_shader(&device, "entrypoint/megakernel.wgsl", &flags)?;

    let scene_bg_layout = scene.make_bind_group_layout(&device);
    let scene_bg = scene.make_bind_group(&device, &queue, &scene_bg_layout);

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

    let mut bg_layouts = vec![&scene_bg_layout, &statics_bg_layout];
    extra_state.add_bind_group_layouts(&mut bg_layouts);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &bg_layouts,
        immediate_size: 4,
    });

    drop(bg_layouts);

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: None,
    });

    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });

    let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
        mapped_at_creation: false,
    });

    let mut last = queue.submit([]);

    for i in options.sample_offset..render_options.samples {
        extra_state.before_sample(i, &device, &queue, &mean, &variance);

        let mut encoder = device.create_command_encoder(&Default::default());

        let begin = (i == options.sample_offset).then_some(0);
        let end = (i + 1 == render_options.samples).then_some(1);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: (begin.is_some() || end.is_some()).then_some(
                    wgpu::ComputePassTimestampWrites {
                        query_set: &query_set,
                        beginning_of_pass_write_index: begin,
                        end_of_pass_write_index: end,
                    },
                ),
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &scene_bg, &[]);
            pass.set_bind_group(1, &statics_bg, &[]);
            pass.set_immediates(0, bytemuck::bytes_of(&i));

            extra_state.setup_pass(&mut pass);

            pass.dispatch_workgroups(
                (render_options.width + 7) / 8,
                (render_options.height + 3) / 4,
                1,
            );
        }

        if end.is_some() {
            encoder.resolve_query_set(&query_set, 0..2, &query_buffer, 0);
        }

        let new = queue.submit([encoder.finish()]);
        device
            .poll(PollType::Wait {
                submission_index: Some(last),
                timeout: None,
            })
            .unwrap();

        last = new;
        eprint!("\r{i}         ");
        std::io::stderr().flush().unwrap();
    }
    eprintln!();

    if std::env::var_os("MESA_VK_TRACE_PER_SUBMIT").is_some() {
        std::thread::sleep(Duration::from_secs(1));
    }

    let mut encoder = device.create_command_encoder(&Default::default());

    let downloaded = Arc::new(Mutex::new((0.0, vec![], vec![])));

    let ns_per_tick = queue.get_timestamp_period();
    let dl = downloaded.clone();
    download_buffer(&device, &mut encoder, &query_buffer, move |data| {
        let data: &[u64] = bytemuck::cast_slice(&data);

        let ticks = data[1] - data[0];
        dl.lock().unwrap().0 = ticks as f64 * ns_per_tick as f64 * 1e-9;
    });

    let dl = downloaded.clone();
    download_texture(&device, &mut encoder, &mean, move |data| {
        dl.lock().unwrap().1 = data;
    });

    let dl = downloaded.clone();
    download_texture(&device, &mut encoder, &variance, move |data| {
        dl.lock().unwrap().2 = data;
    });

    queue.submit([encoder.finish()]);

    device.poll(wgpu::PollType::wait_indefinitely())?;

    let (time, mean, variance) = Arc::into_inner(downloaded).unwrap().into_inner().unwrap();

    let mut avg_rel_var = 0.0;
    let mut avg_rel_err = 0.0;
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

        avg_rel_var += rel_var.element_sum() / 3.0;
        avg_rel_err += rel_err.element_sum() / 3.0;
    }
    let avg_rel_var = avg_rel_var / mean.len() as f32;
    let avg_rel_err = avg_rel_err / mean.len() as f32;

    let avg_sample_time = time as f32 / (render_options.samples - options.sample_offset) as f32;

    println!(
        "Took {time:.2} seconds ({:.3?} / sample)",
        Duration::from_secs_f32(avg_sample_time)
    );
    println!("Average relative variance: {avg_rel_var}");
    println!("Average relative error: {}", avg_rel_err.sqrt());
    println!("Efficiency: {}", 1.0 / (avg_rel_var * avg_sample_time));

    const SRGB_TO_XYZ_T: Mat3 = Mat3::from_cols_array_2d(&[
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ]);
    let xyz_to_srgb = SRGB_TO_XYZ_T.transpose().inverse();

    let mut invalid_pixel = None;

    image::RgbImage::from_vec(
        render_options.width,
        render_options.height,
        mean.into_iter()
            .enumerate()
            .inspect(|&(i, raw)| {
                if !raw.is_finite() {
                    invalid_pixel = Some(i);
                }
            })
            .map(|(_, xyza)| xyz_to_srgb * xyza.xyz() * options.scale)
            .map(|rgb| {
                let low = rgb * 12.92;
                let high = rgb.powf(1.0 / 2.4) * 1.055 - 0.055;
                Vec3::select(rgb.cmplt(Vec3::splat(0.0031308)), low, high)
            })
            .map(|rgb| (rgb * 255.0).as_u8vec3())
            .flat_map(|rgb| rgb.to_array())
            .collect(),
    )
    .unwrap()
    .save("img.png")
    .unwrap();

    if let Some(i) = invalid_pixel {
        println!("Warning: Pixel {i} had non-finite value");
    }

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
}

trait ExtraState {
    fn add_bind_group_layouts<'a>(&'a mut self, bg_layouts: &mut Vec<&'a wgpu::BindGroupLayout>);
    fn setup_pass(&mut self, pass: &mut wgpu::ComputePass);
    fn before_sample(
        &mut self,
        sample: u32,
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
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _mean: &wgpu::Texture,
        _variance: &wgpu::Texture,
    ) {
    }
}

struct GuidedState {
    bsp: wgpu::Buffer,
    dir_tree: wgpu::Buffer,
    bounds: wgpu::Buffer,
    bg_layout: wgpu::BindGroupLayout,
    bg: wgpu::BindGroup,
    iter: u32,
    next_iter: u32,
    no_iter_after_sample: u32,
}

#[derive(Copy, Clone, Debug, NoUninit, AnyBitPattern)]
#[repr(C)]
struct BspNode {
    is_leaf: u32,
    left: u32,
    right: u32,
    count: u32,
}

#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
struct DirTreeNode {
    flux: f32,
    child: u32,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
struct SceneBounds {
    min: Vec3,
    _padding0: u32,
    max: Vec3,
    _padding1: u32,
}

impl ExtraState for GuidedState {
    fn add_bind_group_layouts<'a>(&'a mut self, bg_layouts: &mut Vec<&'a wgpu::BindGroupLayout>) {
        bg_layouts.push(&self.bg_layout);
    }

    fn setup_pass(&mut self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(2, &self.bg, &[]);
    }

    fn before_sample(
        &mut self,
        sample: u32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mean: &wgpu::Texture,
        variance: &wgpu::Texture,
    ) {
        if sample == self.next_iter && sample < self.no_iter_after_sample {
            self.iter += 1;
            self.next_iter += Self::INITIAL_SAMPLES << self.iter;
            println!("\rUpdating guidance model at sample {sample}");

            let bsp = Arc::new(OnceLock::new());
            let bsp2 = bsp.clone();
            wgpu::util::DownloadBuffer::read_buffer(
                device,
                queue,
                &self.bsp.slice(..),
                move |result| {
                    bsp2.set(bytemuck::pod_collect_to_vec(&result.unwrap()))
                        .unwrap();
                },
            );

            let dir_tree = Arc::new(OnceLock::new());
            let dir_tree2 = dir_tree.clone();
            wgpu::util::DownloadBuffer::read_buffer(
                device,
                queue,
                &self.dir_tree.slice(..),
                move |result| {
                    dir_tree2
                        .set(bytemuck::pod_collect_to_vec(&result.unwrap()))
                        .unwrap();
                },
            );

            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

            let mut bsp = Arc::into_inner(bsp).unwrap().into_inner().unwrap();
            let dir_tree = Arc::into_inner(dir_tree).unwrap().into_inner().unwrap();

            let mut new_dir_tree = vec![];

            let split_threshold = Self::C * (1u32 << self.iter).isqrt();

            Self::refine_bsp(&mut bsp, &dir_tree, &mut new_dir_tree, split_threshold, 0);

            self.bsp = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&bsp),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            });

            let train = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&new_dir_tree),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            });
            let guide = std::mem::replace(&mut self.dir_tree, train);

            self.bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.bsp.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: guide.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.dir_tree.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.bounds.as_entire_binding(),
                    },
                ],
            });

            let mut cmd = device.create_command_encoder(&Default::default());
            cmd.clear_texture(mean, &wgpu::ImageSubresourceRange::default());
            cmd.clear_texture(variance, &wgpu::ImageSubresourceRange::default());
            queue.submit([cmd.finish()]);
        }
    }
}

impl GuidedState {
    const LEAF_ENERGY_PORTION: f32 = 0.01;
    const C: u32 = 12000;
    const INITIAL_SAMPLES: u32 = 4;

    fn new(device: &wgpu::Device, scene: &Scene, samples: u32) -> Self {
        let mut qt_nodes = vec![];
        let initial_dir_tree = Self::refine_quadtree(&mut qt_nodes, &[], !0, 1.0, 0);

        let bsp = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&BspNode {
                is_leaf: 1,
                left: !0,
                right: initial_dir_tree,
                count: 0,
            }),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        });

        let initial_guide = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0; std::mem::size_of::<[DirTreeNode; 4]>()],
            usage: wgpu::BufferUsages::STORAGE,
        });

        let initial_train = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(qt_nodes.as_flattened()),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        });

        let bounds = scene.node_bounds(scene.root.unwrap());
        let bounds = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&SceneBounds {
                min: bounds.min,
                max: bounds.max,
                _padding0: 0,
                _padding1: 0,
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                writable_storage_buffer_entry(0),
                storage_buffer_entry(1),
                writable_storage_buffer_entry(2),
                storage_buffer_entry(3),
            ],
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bsp.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: initial_guide.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: initial_train.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bounds.as_entire_binding(),
                },
            ],
        });

        GuidedState {
            bsp,
            dir_tree: initial_train,
            bounds,
            bg_layout,
            bg,
            iter: 0,
            next_iter: Self::INITIAL_SAMPLES,
            no_iter_after_sample: samples * 15 / 100,
        }
    }

    fn refine_quadtree(
        new_nodes: &mut Vec<[DirTreeNode; 4]>,
        existing_nodes: &[[DirTreeNode; 4]],
        node: u32,
        flux_ratio: f32,
        depth: u32,
    ) -> u32 {
        assert!(flux_ratio <= 1.0 && flux_ratio >= 0.0, "{flux_ratio}");
        if flux_ratio < Self::LEAF_ENERGY_PORTION || depth >= 20 {
            return !0;
        }

        let children = match node == !0 {
            true => [(!0, 0.25); 4],
            false => {
                let total_flux: f32 = existing_nodes[node as usize].iter().map(|n| n.flux).sum();
                if total_flux == 0.0 {
                    [(!0, 0.25); 4]
                } else {
                    existing_nodes[node as usize].map(|node| (node.child, node.flux / total_flux))
                }
            }
        };

        let new_children = children.map(|(child, portion)| {
            assert!(portion.is_finite(), "{:?}", existing_nodes[node as usize]);
            DirTreeNode {
                flux: 0.0,
                child: Self::refine_quadtree(
                    new_nodes,
                    existing_nodes,
                    child,
                    flux_ratio * portion,
                    depth + 1,
                ),
            }
        });

        let id = new_nodes.len() as u32;
        new_nodes.push(new_children);
        id
    }

    fn refine_bsp(
        bsp: &mut Vec<BspNode>,
        dir_tree: &[[DirTreeNode; 4]],
        new_dir_tree: &mut Vec<[DirTreeNode; 4]>,
        split_threshold: u32,
        node: u32,
    ) {
        let bsp_len = bsp.len() as u32;
        let n = &mut bsp[node as usize];
        if n.is_leaf == 0 {
            let left = n.left;
            let right = n.right;
            Self::refine_bsp(bsp, dir_tree, new_dir_tree, split_threshold, left);
            Self::refine_bsp(bsp, dir_tree, new_dir_tree, split_threshold, right);
            return;
        }

        if n.count > split_threshold {
            let guide_dt = n.left;
            let train_dt = n.right;
            let count = n.count / 2;
            n.left = bsp_len;
            n.right = bsp_len + 1;
            n.is_leaf = 0;

            bsp.push(BspNode {
                is_leaf: 1,
                left: guide_dt,
                right: train_dt,
                count,
            });
            bsp.push(BspNode {
                is_leaf: 1,
                left: guide_dt,
                right: train_dt,
                count,
            });

            Self::refine_bsp(bsp, dir_tree, new_dir_tree, split_threshold, bsp_len);
            Self::refine_bsp(bsp, dir_tree, new_dir_tree, split_threshold, bsp_len + 1);
            return;
        }

        n.left = n.right;
        n.right = Self::refine_quadtree(new_dir_tree, dir_tree, n.right, 1.0, 0);
        n.count = 0;
    }

    fn output_dirtree(dir_tree: &[[DirTreeNode; 4]], node: u32) {
        fn height(dt: &[[DirTreeNode; 4]], node: u32) -> u32 {
            match node == !0 {
                true => 0,
                false => {
                    1 + dt[node as usize]
                        .iter()
                        .map(|n| height(dt, n.child))
                        .max()
                        .unwrap()
                }
            }
        }
        let resolution = 1 << height(dir_tree, node);

        fn flux_density(dt: &[[DirTreeNode; 4]], node: u32, pos: glam::Vec2, depth: u32) -> f32 {
            let child = pos.cmpge(Vec2::splat(0.5)).bitmask() as usize;
            let child = &dt[node as usize][child];
            if child.child == !0 {
                return child.flux * (1 << 2 * depth) as f32;
            }
            flux_density(dt, child.child, (pos * 2.0).fract(), depth + 1)
        }

        let img = image::ImageBuffer::from_fn(resolution, resolution, |x, y| {
            image::Luma([flux_density(
                dir_tree,
                node,
                Vec2::new(x as f32 + 0.5, y as f32 + 0.5) / resolution as f32,
                0,
            )])
        });
        let max = *img.iter().max_by_key(|&&x| OrderedFloat(x)).unwrap();
        let img = RgbImage::from_fn(resolution, resolution, |x, y| {
            Rgb([(img.get_pixel(x, y).0[0] / max * 255.0) as u8; 3])
        });
        img.save("dirtree.png").unwrap();
    }
}
