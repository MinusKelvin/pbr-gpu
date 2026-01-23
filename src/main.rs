use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use clap::Parser;
use glam::{Mat3, Mat4, Vec4, Vec4Swizzles};
use wgpu::PollType;
use wgpu::util::DeviceExt;

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

    #[clap(short, long, default_value = "128")]
    samples: u32,

    scene: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let options = Options::parse();

    let (mut render_options, scene) = loader::pbrt::load_pbrt_scene(&options.scene);

    if let Some(width) = options.width {
        render_options.width = width;
    }
    if let Some(height) = options.height {
        render_options.height = height;
    }

    let instance = wgpu::Instance::new(&Default::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::SHADER_INT64
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::IMMEDIATES,
        required_limits: wgpu::Limits {
            max_immediate_size: 64,
            max_storage_buffer_binding_size: (2 << 30) - 4,
            max_buffer_size: (2 << 30) - 4,
            ..wgpu::Limits::default().using_resolution(adapter.limits())
        },
        ..Default::default()
    }))?;

    let flags = [
        ("sampler".to_owned(), "independent".to_owned()),
        ("camera".to_owned(), "projective".to_owned()),
        ("integrator".to_owned(), "randomwalk".to_owned()),
    ]
    .into_iter()
    .collect();
    let shader = shader::load_shader(&device, "entrypoint/megakernel.wgsl", &flags)?;

    let scene_bg = scene.make_bind_group(&device);

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
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
    });

    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&render_options.camera),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let spectra_buffer = spectrum::load_spectrums(&device);

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
            storage_buffer_entry(1),
            storage_buffer_entry(2),
        ],
    });

    let statics_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &statics_bg_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &target.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: spectra_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&scene::bind_group_layout(&device), &statics_bg_layout],
        immediate_size: 4,
    });

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

    for i in 0u32..options.samples {
        let mut encoder = device.create_command_encoder(&Default::default());

        {
            let begin = (i == 0).then_some(0);
            let end = (i + 1 == options.samples).then_some(1);
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
            pass.dispatch_workgroups(
                (render_options.width + 7) / 8,
                (render_options.height + 3) / 4,
                1,
            );
        }

        encoder.resolve_query_set(&query_set, 0..2, &query_buffer, 0);

        let new = queue.submit([encoder.finish()]);
        device
            .poll(PollType::Wait {
                submission_index: Some(last),
                timeout: None,
            })
            .unwrap();
        last = new;
        print!("\r{i}         ");
        std::io::stdout().flush().unwrap();
    }
    println!();

    if std::env::var_os("MESA_VK_TRACE_PER_SUBMIT").is_some() {
        std::thread::sleep(Duration::from_secs(1));
    }

    let mut encoder = device.create_command_encoder(&Default::default());

    let ns_per_tick = queue.get_timestamp_period();
    download_buffer(&device, &mut encoder, &query_buffer, move |data| {
        let data: &[u64] = bytemuck::cast_slice(&data);

        let ticks = data[1] - data[0];
        println!(
            "Took {:.3?}",
            Duration::from_secs_f64(ticks as f64 * ns_per_tick as f64 * 1e-9)
        );
    });

    download_texture(&device, &mut encoder, &target, move |data| {
        const SRGB_TO_XYZ_T: Mat3 = Mat3::from_cols_array_2d(&[
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ]);
        let xyz_to_srgb = SRGB_TO_XYZ_T.transpose().inverse();

        image::RgbImage::from_vec(
            render_options.width,
            render_options.height,
            data.into_iter()
                .map(Vec4::from_array)
                .map(|xyza| xyz_to_srgb * xyza.xyz())
                .map(|rgb| (rgb * 255.0).as_u8vec3())
                .flat_map(|rgb| rgb.to_array())
                .collect(),
        )
        .unwrap()
        .save("img.png")
        .unwrap();
    });

    queue.submit([encoder.finish()]);

    device.poll(wgpu::PollType::wait_indefinitely())?;

    Ok(())
}

fn download_texture(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    downloaded: impl FnOnce(Vec<[f32; 4]>) + Send + 'static,
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
        let data: &[[f32; 4]] = bytemuck::cast_slice(&data);
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
