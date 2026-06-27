use std::io::prelude::Write;
use std::time::{Duration, Instant};

use crate::options::RenderOptions;
use crate::scene::Scene;
use crate::shader::load_shader;
use crate::{Options, writable_storage_buffer_entry};

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
    let flags = [
        ("sampler".to_owned(), "independent".to_owned()),
        ("camera".to_owned(), "projective".to_owned()),
    ]
    .into_iter()
    .collect();

    let rays = (render_options.width * render_options.height).next_multiple_of(32);

    let ray_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ray state"),
        size: rays as u64 * 128,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let state_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[writable_storage_buffer_entry(0)],
    });

    let state_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &state_bg_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: ray_state_buffer.as_entire_binding(),
        }],
    });

    let scene_bg_layout = scene.make_bind_group_layout(&device);
    let scene_bg = scene.make_bind_group(&device, &queue, &scene_bg_layout);

    let generated = scene.generated_texture_shader_code();

    let raygen_shader = load_shader("wavefront/raygen.wgsl", &flags)? + &generated;
    let raygen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(raygen_shader.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&scene_bg_layout, &statics_bg_layout, &state_bg_layout],
        immediate_size: 4,
    });

    let raygen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("raygen"),
        layout: Some(&pipeline_layout),
        module: &raygen_shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: None,
    });

    let pathtrace_shader = load_shader("wavefront/pathtrace.wgsl", &flags)? + &generated;
    let pathtrace_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(pathtrace_shader.into()),
    });

    let pathtrace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pathtrace"),
        layout: Some(&pipeline_layout),
        module: &pathtrace_shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: None,
    });

    let add_sample_shader = load_shader("wavefront/add_sample.wgsl", &flags)? + &generated;
    let add_sample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(add_sample_shader.into()),
    });

    let add_sample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("add sample"),
        layout: Some(&pipeline_layout),
        module: &add_sample_shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: None,
    });

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

            pass.set_pipeline(&raygen_pipeline);
            pass.set_immediates(0, bytemuck::bytes_of(&i));
            pass.dispatch_workgroups(
                (render_options.width + 7) / 8,
                (render_options.height + 3) / 4,
                1,
            );

            pass.set_pipeline(&pathtrace_pipeline);
            pass.dispatch_workgroups(rays / 32, 1, 1);

            pass.set_pipeline(&add_sample_pipeline);
            pass.dispatch_workgroups(rays / 32, 1, 1);
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
