use glam::Mat4;

use crate::{ProjectiveCamera, Transform};

pub struct RenderOptions {
    pub camera: ProjectiveCamera,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            camera: ProjectiveCamera {
                ndc_to_camera: Transform::from_mat4_inverse(Mat4::perspective_infinite_lh(
                    90f32.to_radians(),
                    2.0,
                    0.01,
                )),
                world_to_camera: Transform::from_mat4(Mat4::IDENTITY),
                lens_radius: 0.0,
                focal_distance: 1e30,
                orthographic: false as u32,
                _padding: 0,
            },
            width: 1280,
            height: 720,
            samples: 16,
        }
    }
}
