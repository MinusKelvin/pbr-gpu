use bytemuck::NoUninit;

use crate::scene::{Scene, TextureId};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, NoUninit)]
#[repr(C)]
pub struct MaterialId(u32);

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
enum MaterialType {
    Diffuse = 0 << MaterialId::TAG_SHIFT,
    DiffuseTransmit = 1 << MaterialId::TAG_SHIFT,
}

impl MaterialId {
    const TAG_BITS: u32 = 1;
    const TAG_SHIFT: u32 = 32 - Self::TAG_BITS;
    const IDX_MASK: u32 = (1 << Self::TAG_SHIFT) - 1;
    const TAG_MASK: u32 = !Self::IDX_MASK;

    fn new(ty: MaterialType, idx: usize) -> Self {
        assert!(
            idx <= Self::IDX_MASK as usize,
            "cannot exceed {} {ty:?} shapes",
            Self::IDX_MASK
        );
        MaterialId(idx as u32 | ty as u32)
    }

    fn ty(self) -> MaterialType {
        unsafe { std::mem::transmute(self.0 & Self::TAG_MASK) }
    }

    fn idx(self) -> usize {
        (self.0 & Self::IDX_MASK) as usize
    }
}

impl Scene {
    pub fn add_diffuse_material(&mut self, texture: TextureId) -> MaterialId {
        let id = MaterialId::new(MaterialType::Diffuse, self.diffuse_mat.len());
        self.diffuse_mat.push(DiffuseMaterial { texture });
        id
    }

    pub fn add_diffuse_transmit_material(
        &mut self,
        reflectance: TextureId,
        transmittance: TextureId,
    ) -> MaterialId {
        let id = MaterialId::new(MaterialType::DiffuseTransmit, self.diffuse_transmit_mat.len());
        self.diffuse_transmit_mat.push(DiffuseTransmitMaterial {
            reflectance,
            transmittance,
        });
        id
    }
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct DiffuseMaterial {
    pub texture: TextureId,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct DiffuseTransmitMaterial {
    pub reflectance: TextureId,
    pub transmittance: TextureId,
}
