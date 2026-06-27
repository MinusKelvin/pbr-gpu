use std::io::prelude::Write;
use std::time::{Duration, Instant};

use crate::guided_state::GuidedState;
use crate::options::RenderOptions;
use crate::scene::Scene;
use crate::shader::load_shader;
use crate::{ExtraState, Options};

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
    let mut extra_state = match options.integrator.as_str() {
        "guided" => Box::new(GuidedState::new(
            &device,
            &scene,
            options.scale,
            render_options.samples,
            Duration::MAX,
        )) as Box<dyn ExtraState>,
        _ => Box::new(()),
    };

    let flags = [
        ("sampler".to_owned(), options.sampler.clone()),
        ("camera".to_owned(), "projective".to_owned()),
        ("integrator".to_owned(), options.integrator.clone()),
    ]
    .into_iter()
    .collect();
    let shader =
        load_shader("entrypoint/megakernel.wgsl", &flags)? + &scene.generated_texture_shader_code();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });

    let scene_bg_layout = scene.make_bind_group_layout(&device);
    let scene_bg = scene.make_bind_group(&device, &queue, &scene_bg_layout);

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

    let mut last = queue.submit([]);

    let start = Instant::now();
    let mut num_samples = 0;

    for i in options.sample_offset..render_options.samples {
        let time = start.elapsed();
        if start.elapsed() >= render_options.time {
            break;
        }

        num_samples += 1;

        extra_state.before_sample(i, time, &device, &queue, &mean, &variance);

        let mut encoder = device.create_command_encoder(&Default::default());

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &scene_bg, &[]);
            pass.set_bind_group(1, statics_bg, &[]);
            pass.set_immediates(0, bytemuck::bytes_of(&i));

            extra_state.setup_pass(&mut pass);

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
