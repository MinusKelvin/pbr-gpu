use std::path::Path;

use bytemuck::NoUninit;
use glam::Vec3;
use image::DynamicImage;

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
    ImageRgb = 3 << TextureId::TAG_SHIFT,
    Scale = 4 << TextureId::TAG_SHIFT,
    Mix = 5 << TextureId::TAG_SHIFT,
    Checkerboard = 6 << TextureId::TAG_SHIFT,
}

impl TextureId {
    const TAG_BITS: u32 = 3;
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

    pub fn add_image_texture(&mut self, image: u32) -> TextureId {
        let id = TextureId::new(TextureType::ImageRgb, self.image_rgb_tex.len());
        self.image_rgb_tex.push(ImageRgbTexture { image });
        id
    }

    pub fn add_scale_texture(&mut self, left: TextureId, right: TextureId) -> TextureId {
        let id = TextureId::new(TextureType::Scale, self.scale_tex.len());
        self.scale_tex.push(ScaleTexture { left, right });
        id
    }

    pub fn add_mix_texture(
        &mut self,
        tex1: TextureId,
        tex2: TextureId,
        amount: TextureId,
    ) -> TextureId {
        let id = TextureId::new(TextureType::Mix, self.mix_tex.len());
        self.mix_tex.push(MixTexture { tex1, tex2, amount });
        id
    }

    pub fn add_checkerboard_texture(&mut self, even: TextureId, odd: TextureId) -> TextureId {
        let id = TextureId::new(TextureType::Checkerboard, self.checkerboard_tex.len());
        self.checkerboard_tex
            .push(CheckerboardTexture { even, odd });
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

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct ImageRgbTexture {
    pub image: u32,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct ScaleTexture {
    pub left: TextureId,
    pub right: TextureId,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct MixTexture {
    pub tex1: TextureId,
    pub tex2: TextureId,
    pub amount: TextureId,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct CheckerboardTexture {
    pub even: TextureId,
    pub odd: TextureId,
}
