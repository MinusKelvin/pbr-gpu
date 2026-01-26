use bytemuck::NoUninit;

use crate::Transform;
use crate::scene::{NodeId, Scene, ShapeId, SpectrumId};

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
        self.uniform_lights.push(UniformLight { spectrum });
        id
    }

    pub fn add_image_light(&mut self, light: ImageLight) -> LightId {
        let id = LightId::new(LightType::Image, self.image_lights.len());
        self.infinite_lights.push(id);
        self.image_lights.push(light);
        id
    }

    pub fn add_area_light(&mut self, shape: ShapeId, spectrum: SpectrumId) -> LightId {
        let id = LightId::new(LightType::Area, self.area_lights.len());
        self.area_lights.push(AreaLight {
            spectrum,
            transform_node: NodeId::ZERO,
            shape,
        });
        id
    }

    pub fn set_area_light_transform(&mut self, light: LightId, transform: NodeId) {
        self.area_lights[light.idx()].transform_node = transform;
    }
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct UniformLight {
    pub spectrum: SpectrumId,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct ImageLight {
    pub transform: Transform,
    pub image: u32,
    pub scale: f32,
    pub _padding: [u32; 2],
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct AreaLight {
    pub spectrum: SpectrumId,
    pub transform_node: NodeId,
    pub shape: ShapeId,
}
