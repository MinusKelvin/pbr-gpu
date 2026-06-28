use std::io::prelude::Write;
use std::time::{Duration, Instant};

use wgpu::{BindGroupDescriptor, BindGroupLayoutDescriptor};

use crate::options::RenderOptions;
use crate::scene::Scene;
use crate::shader::load_shader;
use crate::{Options, storage_buffer_entry, writable_storage_buffer_entry};

pub fn run(
    options: &Options,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &Scene,
    render_options: RenderOptions,
    statics_bg_layout: &wgpu::BindGroupLayout,
    statics_bg: &wgpu::BindGroup,
    mean: &wgpu::Texture,
    variance: &wgpu::Texture,
) -> anyhow::Result<(u32, Duration)> {
    let rays = render_options.width * render_options.height;
    let wg_size = (rays + 31) / 32;

    let ray_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ray state"),
        size: rays as u64 * 80,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let path_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("path state"),
        size: rays as u64 * 32,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let surface_hit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("surface hit state"),
        size: rays as u64 * 128,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let shadow_ray_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shadow ray state"),
        size: rays as u64 * 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let state_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            writable_storage_buffer_entry(0),
            writable_storage_buffer_entry(1),
            writable_storage_buffer_entry(2),
            writable_storage_buffer_entry(3),
        ],
    });

    let state_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &state_bg_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ray_state_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: path_state_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: surface_hit_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: shadow_ray_buffer.as_entire_binding(),
            },
        ],
    });

    let trace_queue = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 + rays as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let direct_light_queue = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 + rays as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bounce_queue = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 + rays as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let shadow_queue = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 + rays as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let queue_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            writable_storage_buffer_entry(0),
            writable_storage_buffer_entry(1),
            writable_storage_buffer_entry(2),
            writable_storage_buffer_entry(3),
        ],
    });

    let queue_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &queue_bg_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: trace_queue.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: direct_light_queue.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bounce_queue.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: shadow_queue.as_entire_binding(),
            },
        ],
    });

    let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64,
        usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let indirect_bg_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[writable_storage_buffer_entry(0)],
    });

    let indirect_bg = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &indirect_bg_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: indirect_buffer.as_entire_binding(),
        }],
    });

    let scene_bg_layout = scene.make_bind_group_layout(&device);
    let scene_bg = scene.make_bind_group(&device, &queue, &scene_bg_layout);

    let generated = scene.generated_texture_shader_code();

    let common_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[
            Some(&scene_bg_layout),
            Some(&statics_bg_layout),
            Some(&state_bg_layout),
            Some(&queue_bg_layout),
        ],
        immediate_size: 4,
    });

    let prep_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[
            None,
            None,
            None,
            Some(&queue_bg_layout),
            Some(&indirect_bg_layout),
        ],
        immediate_size: 4,
    });

    let raygen = make_pipeline("raygen", device, &common_layout, &generated)?;
    let trace_ray = make_pipeline("trace_ray", device, &common_layout, &generated)?;
    let sample_light = make_pipeline("sample_light", device, &common_layout, &generated)?;
    let shadow_ray = make_pipeline("shadow_ray", device, &common_layout, &generated)?;
    let bounce = make_pipeline("bounce", device, &common_layout, &generated)?;
    let add_sample = make_pipeline("add_sample", device, &common_layout, &generated)?;
    let prep_shading = make_pipeline("prep_shading", device, &prep_layout, "")?;
    let prep_tracing = make_pipeline("prep_tracing", device, &prep_layout, "")?;
    let prep_shadow = make_pipeline("prep_shadow", device, &prep_layout, "")?;

    let mut last = queue.submit([]);

    let start = Instant::now();
    let mut num_samples = 0;

    for i in options.sample_offset..render_options.samples {
        if start.elapsed() >= render_options.time {
            break;
        }

        num_samples += 1;

        let mut encoder = device.create_command_encoder(&Default::default());

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            pass.set_bind_group(0, &scene_bg, &[]);
            pass.set_bind_group(1, statics_bg, &[]);
            pass.set_bind_group(2, &state_bg, &[]);
            pass.set_bind_group(3, &queue_bg, &[]);
            pass.set_bind_group(4, &indirect_bg, &[]);

            pass.set_pipeline(&prep_shading);
            pass.dispatch_workgroups(1, 1, 1);

            pass.set_pipeline(&raygen);
            pass.set_immediates(0, bytemuck::bytes_of(&i));
            pass.dispatch_workgroups(
                (render_options.width + 7) / 8,
                (render_options.height + 7) / 8,
                1,
            );

            for i in 0..32 {
                pass.push_debug_group(&format!("bounce {i}"));

                pass.set_pipeline(&prep_tracing);
                pass.dispatch_workgroups(1, 1, 1);

                pass.set_pipeline(&trace_ray);
                pass.dispatch_workgroups_indirect(&indirect_buffer, 0);

                pass.set_pipeline(&prep_shading);
                pass.dispatch_workgroups(1, 1, 1);

                pass.set_pipeline(&sample_light);
                pass.dispatch_workgroups_indirect(&indirect_buffer, 16);

                pass.set_pipeline(&prep_shadow);
                pass.dispatch_workgroups(1, 1, 1);

                pass.set_pipeline(&shadow_ray);
                pass.dispatch_workgroups_indirect(&indirect_buffer, 48);

                pass.set_pipeline(&bounce);
                pass.dispatch_workgroups_indirect(&indirect_buffer, 32);

                pass.pop_debug_group();
            }

            pass.set_pipeline(&prep_tracing);
            pass.dispatch_workgroups(1, 1, 1);

            pass.set_pipeline(&trace_ray);
            pass.dispatch_workgroups_indirect(&indirect_buffer, 0);

            pass.set_pipeline(&add_sample);
            pass.dispatch_workgroups(
                (render_options.width + 7) / 8,
                (render_options.height + 7) / 8,
                1,
            );
        }

        let new = queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(last),
                timeout: None,
            })
            .unwrap();

        last = new;
        eprint!("\r{}         ", i + 1);
        std::io::stderr().flush().unwrap();
    }
    eprintln!();

    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(last),
            timeout: None,
        })
        .unwrap();

    Ok((num_samples, start.elapsed()))
}

fn make_pipeline(
    name: &str,
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    generated: &str,
) -> anyhow::Result<wgpu::ComputePipeline> {
    let flags = [
        ("sampler".to_owned(), "independent".to_owned()),
        ("camera".to_owned(), "projective".to_owned()),
    ]
    .into_iter()
    .collect();

    let shader = load_shader(&format!("wavefront/{name}.wgsl"), &flags)? + generated;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });

    Ok(
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&layout),
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        }),
    )
}
