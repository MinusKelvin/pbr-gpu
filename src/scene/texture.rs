use bytemuck::NoUninit;

use crate::scene::{Scene, SpectrumId};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, NoUninit)]
#[repr(C)]
pub struct TextureId(u32);

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
enum TextureType {
    Constant = 0 << TextureId::TAG_SHIFT,
    ImageFloat = 2 << TextureId::TAG_SHIFT,
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
    pub fn add_constant_texture(&mut self, spectrum: SpectrumId) -> TextureId {
        let id = TextureId::new(TextureType::Constant, self.constant_tex.len());
        self.constant_tex.push(ConstantTexture { spectrum });
        id
    }

    pub fn add_rgb_image_texture(&mut self, image: u32) -> TextureId {
        let id = TextureId::new(TextureType::ImageRgb, self.image_rgb_tex.len());
        self.image_rgb_tex.push(ImageRgbTexture { image });
        id
    }

    pub fn add_float_image_texture(&mut self, image: u32) -> TextureId {
        let id = TextureId::new(TextureType::ImageFloat, self.image_rgb_tex.len());
        self.image_float_tex.push(ImageFloatTexture { image });
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
pub struct ConstantTexture {
    pub spectrum: SpectrumId,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct ImageRgbTexture {
    pub image: u32,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct ImageFloatTexture {
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
