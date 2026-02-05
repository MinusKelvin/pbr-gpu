use bytemuck::NoUninit;
use glam::DMat4;

use crate::Transform;
use crate::scene::{NodeId, Scene, ShapeId, SpectrumId, TableSampler2d, TextureId};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, NoUninit)]
#[repr(C)]
pub struct LightId(u32);

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
enum LightType {
    Uniform = 0 << LightId::TAG_SHIFT,
    Image = 1 << LightId::TAG_SHIFT,
    Area = 2 << LightId::TAG_SHIFT,
}

#[allow(unused)]
impl LightId {
    pub const ZERO: LightId = LightId(0);

    const TAG_BITS: u32 = 2;
    const TAG_SHIFT: u32 = 32 - Self::TAG_BITS;
    const IDX_MASK: u32 = (1 << Self::TAG_SHIFT) - 1;
    const TAG_MASK: u32 = !Self::IDX_MASK;

    fn new(ty: LightType, idx: usize) -> Self {
        assert!(
            idx <= Self::IDX_MASK as usize,
            "cannot exceed {} {ty:?} shapes",
            Self::IDX_MASK
        );
        LightId(idx as u32 | ty as u32)
    }

    fn ty(self) -> LightType {
        unsafe { std::mem::transmute(self.0 & Self::TAG_MASK) }
    }

    fn idx(self) -> usize {
        (self.0 & Self::IDX_MASK) as usize
    }
}

impl Scene {
    pub fn add_uniform_light(&mut self, spectrum: SpectrumId) -> LightId {
        let id = LightId::new(LightType::Uniform, self.uniform_lights.len());
        self.infinite_lights.push(id);
        self.all_lights.push(id);
        self.uniform_lights.push(UniformLight {
            spectrum,
            light_sampling_path: 0,
        });
        id
    }

    pub fn add_image_light(&mut self, transform: DMat4, image: u32, scale: f32) -> LightId {
        let sampling_distr = self.image_sampling_distribution(image);

        let id = LightId::new(LightType::Image, self.image_lights.len());
        self.infinite_lights.push(id);
        self.all_lights.push(id);
        self.image_lights.push(ImageLight {
            transform: Transform {
                m: transform.as_mat4(),
                m_inv: transform.inverse().as_mat4(),
            },
            image,
            scale,
            light_sampling_path: 0,
            sampling_distr,
            _padding: [0; 2],
        });
        id
    }

    pub fn add_area_light(
        &mut self,
        shape: ShapeId,
        spectrum: SpectrumId,
        alpha: TextureId,
    ) -> LightId {
        let id = LightId::new(LightType::Area, self.area_lights.len());
        self.all_lights.push(id);
        self.area_lights.push(AreaLight {
            spectrum,
            transform_node: NodeId::ZERO,
            shape,
            alpha,
            two_sided: false as u32,
            light_sampling_path: 0,
        });
        id
    }

    pub fn set_area_light_transform(&mut self, light: LightId, transform: NodeId) {
        self.area_lights[light.idx()].transform_node = transform;
    }

    pub fn set_light_sampling_path(&mut self, light: LightId, path: u32) {
        match light.ty() {
            LightType::Uniform => self.uniform_lights[light.idx()].light_sampling_path = path,
            LightType::Image => self.image_lights[light.idx()].light_sampling_path = path,
            LightType::Area => self.area_lights[light.idx()].light_sampling_path = path,
        }
    }
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct UniformLight {
    pub spectrum: SpectrumId,
    pub light_sampling_path: u32,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct ImageLight {
    pub transform: Transform,
    pub image: u32,
    pub scale: f32,
    pub light_sampling_path: u32,
    pub sampling_distr: TableSampler2d,
    pub _padding: [u32; 2],
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct AreaLight {
    pub spectrum: SpectrumId,
    pub transform_node: NodeId,
    pub shape: ShapeId,
    pub alpha: TextureId,
    pub two_sided: u32,
    pub light_sampling_path: u32,
}
