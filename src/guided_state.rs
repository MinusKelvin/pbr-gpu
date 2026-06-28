use std::sync::{Arc, OnceLock};
use std::time::Duration;

use bytemuck::{AnyBitPattern, NoUninit, Pod, Zeroable};
use glam::{Vec2, Vec3};
use image::{Rgb, RgbImage};
use ordered_float::OrderedFloat;
use wgpu::util::DeviceExt;

use crate::scene::Scene;
use crate::{ExtraState, storage_buffer_entry, writable_storage_buffer_entry};

pub struct GuidedState {
    bsp: wgpu::Buffer,
    train_tree: wgpu::Buffer,
    bounds: wgpu::Buffer,
    bg_layout: wgpu::BindGroupLayout,
    bg: wgpu::BindGroup,
    iter: u32,
    next_iter: u32,
    train_budget_samples: u32,
    train_budget_time: Duration,
    scale: f32,
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
struct GuideTreeNode {
    child: u32,
    pr: f32,
}

#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
struct TrainTreeNode {
    child: u32,
    sum: f32,
    comp: f32,
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
    fn add_bind_group_layouts<'a>(
        &'a mut self,
        bg_layouts: &mut Vec<Option<&'a wgpu::BindGroupLayout>>,
    ) {
        bg_layouts.push(Some(&self.bg_layout));
    }

    fn setup_pass(&mut self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(2, &self.bg, &[]);
    }

    fn before_sample(
        &mut self,
        sample: u32,
        time: Duration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mean: &wgpu::Texture,
        variance: &wgpu::Texture,
    ) {
        if sample == self.next_iter
            && sample < self.train_budget_samples
            && time < self.train_budget_time
        {
            self.iter += 1;
            self.next_iter += Self::INITIAL_SAMPLES << self.iter;
            println!("\rUpdating guidance model at sample {sample}");

            let stats = super::collect_stats(device, queue, mean, variance, time);
            println!("Average variance: {}", stats.avg_variance);
            println!("Relative variance: {}", stats.avg_rel_variance);

            let preview_path = format!("preview-{}.png", self.iter);
            super::xyz_to_srgb(&stats.mean_image, self.scale)
                .save(&preview_path)
                .unwrap();
            std::fs::copy(&preview_path, "img.png").unwrap();

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

            let old_train_tree = Arc::new(OnceLock::new());
            let old_train_tree2 = old_train_tree.clone();
            wgpu::util::DownloadBuffer::read_buffer(
                device,
                queue,
                &self.train_tree.slice(..),
                move |result| {
                    old_train_tree2
                        .set(bytemuck::pod_collect_to_vec(&result.unwrap()))
                        .unwrap();
                },
            );

            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

            let mut bsp = Arc::into_inner(bsp).unwrap().into_inner().unwrap();
            let old_train_tree = Arc::into_inner(old_train_tree)
                .unwrap()
                .into_inner()
                .unwrap();

            let mut new_train_tree = vec![];
            let mut new_guide_tree = old_train_tree
                .iter()
                .map(|n: &[TrainTreeNode; 4]| {
                    n.map(|n| GuideTreeNode {
                        child: n.child,
                        pr: 0.0,
                    })
                })
                .collect::<Vec<_>>();

            let split_threshold = Self::C * (1u32 << self.iter).isqrt();

            refine_bsp(
                &mut bsp,
                &old_train_tree,
                &mut new_guide_tree,
                &mut new_train_tree,
                split_threshold,
                0,
            );

            self.bsp.destroy();

            self.bsp = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&bsp),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            });

            self.train_tree = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&new_train_tree),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            });

            let guide = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&new_guide_tree),
                usage: wgpu::BufferUsages::STORAGE,
            });

            // std::fs::write("bsp.dat", bytemuck::cast_slice(&bsp)).unwrap();
            // std::fs::write("guide.dat", bytemuck::cast_slice(&new_guide_tree)).unwrap();
            // std::fs::write("train.dat", bytemuck::cast_slice(&new_train_tree)).unwrap();

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
                        resource: self.train_tree.as_entire_binding(),
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
    const C: u32 = 32000;
    const INITIAL_SAMPLES: u32 = 4;

    pub fn new(
        device: &wgpu::Device,
        scene: &Scene,
        scale: f32,
        samples: u32,
        time: Duration,
    ) -> Self {
        let mut qt_nodes = vec![];
        let mut initial_bsp = vec![BspNode {
            is_leaf: 1,
            left: !0,
            right: !0,
            count: 8 * 8,
        }];
        refine_bsp(&mut initial_bsp, &[], &mut [], &mut qt_nodes, 0, 0);

        // let initial_bsp: Vec<BspNode> =
        //     bytemuck::pod_collect_to_vec(&std::fs::read("bsp.dat").unwrap());
        // let initial_guide: Vec<[GuideTreeNode; 4]> =
        //     bytemuck::pod_collect_to_vec(&std::fs::read("guide.dat").unwrap());
        // let qt_nodes: Vec<[TrainTreeNode; 4]> =
        //     bytemuck::pod_collect_to_vec(&std::fs::read("train.dat").unwrap());

        let bsp = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&initial_bsp),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        });

        let initial_guide = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0; std::mem::size_of::<[GuideTreeNode; 4]>()],
            // contents: bytemuck::cast_slice(&initial_guide),
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
            train_tree: initial_train,
            bounds,
            bg_layout,
            bg,
            iter: 0,
            next_iter: Self::INITIAL_SAMPLES,
            train_budget_samples: (samples as f64 * 0.15) as u32,
            train_budget_time: time.mul_f64(0.15),
            scale,
        }
    }
}

fn normalize_quadtree(
    result: &mut [[GuideTreeNode; 4]],
    train: &[[TrainTreeNode; 4]],
    node: u32,
    size: f32,
) -> f32 {
    assert_ne!(node, !0);

    let children_values = train[node as usize].map(|n| match n.child == !0 {
        true => ((n.sum + n.comp) * size * size).sqrt(),
        false => normalize_quadtree(result, train, n.child, size * 0.5),
    });

    let total: f32 = children_values.iter().sum();

    assert!(total.is_finite(), "{total} {:?}", train[node as usize]);

    for (result, value) in result[node as usize].iter_mut().zip(children_values) {
        result.pr = match total == 0.0 {
            true => 0.25,
            false => value / total,
        }
    }

    total
}

fn refine_quadtree(
    new_nodes: &mut Vec<[TrainTreeNode; 4]>,
    existing_nodes: &[[GuideTreeNode; 4]],
    node: u32,
    flux_ratio: f32,
    depth: u32,
) -> u32 {
    assert!(flux_ratio <= 1.0 && flux_ratio >= 0.0, "{flux_ratio}");
    if flux_ratio < GuidedState::LEAF_ENERGY_PORTION || depth >= 20 {
        return !0;
    }

    let children = match node == !0 {
        true => {
            [GuideTreeNode {
                child: !0,
                pr: 0.25,
            }; 4]
        }
        false => existing_nodes[node as usize],
    };

    let new_children = children.map(|node| TrainTreeNode {
        sum: 0.0,
        comp: 0.0,
        child: refine_quadtree(
            new_nodes,
            existing_nodes,
            node.child,
            flux_ratio * node.pr,
            depth + 1,
        ),
    });

    let id = new_nodes.len() as u32;
    new_nodes.push(new_children);
    id
}

fn refine_bsp(
    bsp: &mut Vec<BspNode>,
    old_train_tree: &[[TrainTreeNode; 4]],
    new_guide_tree: &mut [[GuideTreeNode; 4]],
    new_train_tree: &mut Vec<[TrainTreeNode; 4]>,
    split_threshold: u32,
    node: u32,
) {
    let bsp_len = bsp.len() as u32;
    let n = &mut bsp[node as usize];
    if n.is_leaf == 0 {
        let left = n.left;
        let right = n.right;
        refine_bsp(
            bsp,
            old_train_tree,
            new_guide_tree,
            new_train_tree,
            split_threshold,
            left,
        );
        refine_bsp(
            bsp,
            old_train_tree,
            new_guide_tree,
            new_train_tree,
            split_threshold,
            right,
        );
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

        refine_bsp(
            bsp,
            old_train_tree,
            new_guide_tree,
            new_train_tree,
            split_threshold,
            bsp_len,
        );
        refine_bsp(
            bsp,
            old_train_tree,
            new_guide_tree,
            new_train_tree,
            split_threshold,
            bsp_len + 1,
        );
        return;
    }

    n.left = n.right;
    if n.right != !0 {
        normalize_quadtree(new_guide_tree, old_train_tree, n.right, 1.0);
    }
    n.right = refine_quadtree(new_train_tree, new_guide_tree, n.right, 1.0, 0);
    n.count = 0;
}

fn output_dirtree(dir_tree: &[[GuideTreeNode; 4]], node: u32) {
    fn height(dt: &[[GuideTreeNode; 4]], node: u32) -> u32 {
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

    fn pr_density(dt: &[[GuideTreeNode; 4]], node: u32, pos: glam::Vec2, depth: u32) -> f32 {
        let child = pos.cmpge(Vec2::splat(0.5)).bitmask() as usize;
        let child = &dt[node as usize][child];
        if child.child == !0 {
            return child.pr * (1 << 2 * depth) as f32;
        }
        pr_density(dt, child.child, (pos * 2.0).fract(), depth + 1)
    }

    let img = image::ImageBuffer::from_fn(resolution, resolution, |x, y| {
        image::Luma([pr_density(
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
