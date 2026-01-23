use bytemuck::NoUninit;
use glam::Vec3;

use crate::scene::Scene;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, NoUninit)]
#[repr(C)]
pub struct TextureId(u32);

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
enum TextureType {
    ConstantFloat = 0 << TextureId::TAG_SHIFT,
    ConstantRgb = 1 << TextureId::TAG_SHIFT,
    ConstantSpectrum = 2 << TextureId::TAG_SHIFT,
}

impl TextureId {
    const TAG_BITS: u32 = 2;
    const TAG_SHIFT: u32 = 32 - Self::TAG_BITS;
    const IDX_MASK: u32 = (1 << Self::TAG_SHIFT) - 1;
    const TAG_MASK: u32 = !Self::IDX_MASK;

    fn new(ty: TextureType, idx: usize) -> Self {
        assert!(
            idx <= Self::IDX_MASK as usize,
            "cannot exceed {} {ty:?} shapes",
            Self::IDX_MASK
        );
        TextureId(idx as u32 | ty as u32)
    }

    fn ty(self) -> TextureType {
        unsafe { std::mem::transmute(self.0 & Self::TAG_MASK) }
    }

    fn idx(self) -> usize {
        (self.0 & Self::IDX_MASK) as usize
    }
}

impl Scene {
    pub fn add_constant_float_texture(&mut self, value: f32) -> TextureId {
        let id = TextureId::new(TextureType::ConstantFloat, self.constant_float_tex.len());
        self.constant_float_tex.push(ConstantFloatTexture { value });
        id
    }

    pub fn add_constant_rgb_texture(&mut self, rgb: Vec3) -> TextureId {
        let id = TextureId::new(TextureType::ConstantRgb, self.constant_rgb_tex.len());
        self.constant_rgb_tex
            .push(ConstantRgbTexture { rgb, _padding: 0 });
        id
    }

    pub fn add_constant_spectrum_texture(&mut self, spectrum: u32) -> TextureId {
        let id = TextureId::new(
            TextureType::ConstantSpectrum,
            self.constant_spectrum_tex.len(),
        );
        self.constant_spectrum_tex
            .push(ConstantSpectrumTexture { spectrum });
        id
    }
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct ConstantFloatTexture {
    pub value: f32,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct ConstantRgbTexture {
    pub rgb: Vec3,
    pub _padding: u32,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct ConstantSpectrumTexture {
    pub spectrum: u32,
}
