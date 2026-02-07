use bytemuck::NoUninit;

use crate::scene::{LightId, Scene};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, NoUninit)]
#[repr(C)]
pub struct LightSamplerId(u32);

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
enum LightSamplerType {
    Uniform = 0 << LightSamplerId::TAG_SHIFT,
    Power = 1 << LightSamplerId::TAG_SHIFT,
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

    pub fn add_power_light_sampler(&mut self, lights: &[LightId]) -> LightSamplerId {
        let mut powers: Vec<_> = lights
            .iter()
            .map(|&light| match light.is_infinite() {
                true => 0.0,
                false => self.light_power(light),
            })
            .collect();
        let total_noninfinite: f32 = powers.iter().sum();
        let mut total = total_noninfinite;

        for (power, &light) in powers.iter_mut().zip(lights) {
            if light.is_infinite() {
                *power = total_noninfinite;
                total += total_noninfinite;
            }
        }

        powers.iter_mut().for_each(|v| *v /= total);
        let pmf = powers;

        let mut under = Vec::with_capacity(lights.len());
        let mut over = Vec::with_capacity(lights.len());
        let mut table = Vec::with_capacity(lights.len());

        for i in 0..lights.len() {
            let remaining = pmf[i] * lights.len() as f32;
            if remaining < 1.0 {
                under.push((i, remaining));
            } else {
                over.push((i, remaining));
            }
            table.push(PlsAliasBucket {
                light: lights[i],
                pmf: pmf[i],
                q: 0.0,
                alias: u32::MAX,
            });
            self.set_light_sampling_path(lights[i], i as u32);
        }

        while !under.is_empty() && !over.is_empty() {
            let (low, low_remain) = under.pop().unwrap();
            let (high, high_remain) = over.pop().unwrap();

            table[low].q = low_remain;
            table[low].alias = high as u32;

            let new_high_remain = high_remain + low_remain - 1.0;
            if new_high_remain < 1.0 {
                under.push((high, new_high_remain));
            } else {
                over.push((high, new_high_remain));
            }
        }

        for (idx, _) in under.into_iter().chain(over) {
            table[idx].q = 1.0;
        }

        let id = LightSamplerId::new(LightSamplerType::Power, self.power_light_samplers.len());

        self.power_light_samplers.push(PowerLightSampler {
            ptr: self.power_light_sampler_data.len() as u32,
            count: lights.len() as u32,
        });
        self.power_light_sampler_data.extend_from_slice(&table);

        id
    }
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct UniformLightSampler {
    ptr: u32,
    count: u32,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct PowerLightSampler {
    ptr: u32,
    count: u32,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct PlsAliasBucket {
    light: LightId,
    pmf: f32,
    q: f32,
    alias: u32,
}
