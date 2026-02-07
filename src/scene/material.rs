use bytemuck::NoUninit;

use crate::scene::{Scene, SpectrumId, TextureId};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, NoUninit)]
#[repr(C)]
pub struct MaterialId(u32);

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
enum MaterialType {
    Diffuse = 0 << MaterialId::TAG_SHIFT,
    DiffuseTransmit = 1 << MaterialId::TAG_SHIFT,
    Conductor = 2 << MaterialId::TAG_SHIFT,
    Dielectric = 3 << MaterialId::TAG_SHIFT,
    ThinDielectric = 4 << MaterialId::TAG_SHIFT,
    MetallicWorkflow = 5 << MaterialId::TAG_SHIFT,
    Mix = 6 << MaterialId::TAG_SHIFT,
}

#[allow(unused)]
impl MaterialId {
    const TAG_BITS: u32 = 3;
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
    pub fn add_diffuse_material(
        &mut self,
        texture: TextureId,
        normal_map: Option<u32>,
    ) -> MaterialId {
        let id = MaterialId::new(MaterialType::Diffuse, self.diffuse_mat.len());
        self.diffuse_mat.push(DiffuseMaterial {
            texture,
            normal_map: normal_map.unwrap_or(u32::MAX),
        });
        id
    }

    pub fn add_diffuse_transmit_material(
        &mut self,
        reflectance: TextureId,
        transmittance: TextureId,
        scale: TextureId,
        normal_map: Option<u32>,
    ) -> MaterialId {
        let id = MaterialId::new(
            MaterialType::DiffuseTransmit,
            self.diffuse_transmit_mat.len(),
        );
        self.diffuse_transmit_mat.push(DiffuseTransmitMaterial {
            reflectance,
            transmittance,
            scale,
            normal_map: normal_map.unwrap_or(u32::MAX),
        });
        id
    }

    pub fn add_conductor_material(
        &mut self,
        ior_re: TextureId,
        ior_im: TextureId,
        u_roughness: TextureId,
        v_roughness: TextureId,
        normal_map: Option<u32>,
    ) -> MaterialId {
        let id = MaterialId::new(MaterialType::Conductor, self.conductor_mat.len());
        self.conductor_mat.push(ConductorMaterial {
            ior_re,
            ior_im,
            u_roughness,
            v_roughness,
            normal_map: normal_map.unwrap_or(u32::MAX),
        });
        id
    }

    pub fn add_dielectric_material(
        &mut self,
        ior: SpectrumId,
        u_roughness: TextureId,
        v_roughness: TextureId,
        normal_map: Option<u32>,
    ) -> MaterialId {
        let id = MaterialId::new(MaterialType::Dielectric, self.dielectric_mat.len());
        self.dielectric_mat.push(DielectricMaterial {
            ior,
            u_roughness,
            v_roughness,
            normal_map: normal_map.unwrap_or(u32::MAX),
        });
        id
    }

    pub fn add_thin_dielectric_material(
        &mut self,
        ior: SpectrumId,
        normal_map: Option<u32>,
    ) -> MaterialId {
        let id = MaterialId::new(MaterialType::ThinDielectric, self.thin_dielectric_mat.len());
        self.thin_dielectric_mat.push(ThinDielectricMaterial {
            ior,
            normal_map: normal_map.unwrap_or(u32::MAX),
        });
        id
    }

    pub fn add_metallic_workflow_material(
        &mut self,
        base_color: TextureId,
        metallic: TextureId,
        u_roughness: TextureId,
        v_roughness: TextureId,
        normal_map: Option<u32>,
    ) -> MaterialId {
        let id = MaterialId::new(
            MaterialType::MetallicWorkflow,
            self.metallic_workflow_mat.len(),
        );
        self.metallic_workflow_mat.push(MetallicWorkflowMaterial {
            base_color,
            metallic,
            u_roughness,
            v_roughness,
            normal_map: normal_map.unwrap_or(u32::MAX),
        });
        id
    }

    pub fn add_mix_material(
        &mut self,
        m1: MaterialId,
        m2: MaterialId,
        amount: TextureId,
    ) -> MaterialId {
        let id = MaterialId::new(MaterialType::Mix, self.mix_mat.len());
        self.mix_mat.push(MixMaterial { m1, m2, amount });
        id
    }
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct DiffuseMaterial {
    pub normal_map: u32,
    pub texture: TextureId,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct DiffuseTransmitMaterial {
    pub normal_map: u32,
    pub reflectance: TextureId,
    pub transmittance: TextureId,
    pub scale: TextureId,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct ConductorMaterial {
    pub normal_map: u32,
    pub ior_re: TextureId,
    pub ior_im: TextureId,
    pub u_roughness: TextureId,
    pub v_roughness: TextureId,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct DielectricMaterial {
    pub normal_map: u32,
    pub ior: SpectrumId,
    pub u_roughness: TextureId,
    pub v_roughness: TextureId,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct ThinDielectricMaterial {
    pub normal_map: u32,
    pub ior: SpectrumId,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct MetallicWorkflowMaterial {
    pub normal_map: u32,
    pub base_color: TextureId,
    pub metallic: TextureId,
    pub u_roughness: TextureId,
    pub v_roughness: TextureId,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct MixMaterial {
    pub m1: MaterialId,
    pub m2: MaterialId,
    pub amount: TextureId,
}
