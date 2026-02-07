use bytemuck::NoUninit;

use crate::scene::{LightId, Scene};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, NoUninit)]
#[repr(C)]
pub struct LightSamplerId(u32);

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
enum LightSamplerType {
    Uniform = 0 << LightSamplerId::TAG_SHIFT,
}

#[allow(unused)]
impl LightSamplerId {
    const TAG_BITS: u32 = 1;
    const TAG_SHIFT: u32 = 32 - Self::TAG_BITS;
    const IDX_MASK: u32 = (1 << Self::TAG_SHIFT) - 1;
    const TAG_MASK: u32 = !Self::IDX_MASK;

    fn new(ty: LightSamplerType, idx: usize) -> Self {
        assert!(
            idx <= Self::IDX_MASK as usize,
            "cannot exceed {} {ty:?} shapes",
            Self::IDX_MASK
        );
        LightSamplerId(idx as u32 | ty as u32)
    }

    fn ty(self) -> LightSamplerType {
        unsafe { std::mem::transmute(self.0 & Self::TAG_MASK) }
    }

    fn idx(self) -> usize {
        (self.0 & Self::IDX_MASK) as usize
    }
}

impl Scene {
    pub fn add_uniform_light_sampler(&mut self, lights: &[LightId]) -> LightSamplerId {
        let id = LightSamplerId::new(LightSamplerType::Uniform, self.uniform_light_samplers.len());

        let ptr = self.uniform_light_sampler_data.len() as u32;
        self.uniform_light_sampler_data.extend_from_slice(lights);

        for (i, &light) in lights.iter().enumerate() {
            self.set_light_sampling_path(light, i as u32);
        }

        self.uniform_light_samplers.push(UniformLightSampler {
            ptr,
            count: lights.len() as u32,
        });

        id
    }
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct UniformLightSampler {
    ptr: u32,
    count: u32,
}
