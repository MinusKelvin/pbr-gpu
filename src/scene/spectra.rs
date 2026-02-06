use bytemuck::NoUninit;
use glam::Vec3;

use crate::scene::Scene;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, NoUninit)]
#[repr(C)]
pub struct SpectrumId(u32);

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
enum SpectrumType {
    Table = 0 << SpectrumId::TAG_SHIFT,
    Constant = 1 << SpectrumId::TAG_SHIFT,
    RgbAlbedo = 2 << SpectrumId::TAG_SHIFT,
    RgbIlluminant = 3 << SpectrumId::TAG_SHIFT,
    Blackbody = 4 << SpectrumId::TAG_SHIFT,
    PiecewiseLinear = 5 << SpectrumId::TAG_SHIFT,
    RgbIorIm = 6 << SpectrumId::TAG_SHIFT,
}

#[allow(unused)]
impl SpectrumId {
    pub const D65: SpectrumId = SpectrumId(3);

    const TAG_BITS: u32 = 3;
    const TAG_SHIFT: u32 = 32 - Self::TAG_BITS;
    const IDX_MASK: u32 = (1 << Self::TAG_SHIFT) - 1;
    const TAG_MASK: u32 = !Self::IDX_MASK;

    fn new(ty: SpectrumType, idx: usize) -> Self {
        assert!(
            idx <= Self::IDX_MASK as usize,
            "cannot exceed {} {ty:?} shapes",
            Self::IDX_MASK
        );
        SpectrumId(idx as u32 | ty as u32)
    }

    fn ty(self) -> SpectrumType {
        unsafe { std::mem::transmute(self.0 & Self::TAG_MASK) }
    }

    fn idx(self) -> usize {
        (self.0 & Self::IDX_MASK) as usize
    }
}

impl Scene {
    pub fn add_table_spectrum(&mut self, spectrum: TableSpectrum) -> SpectrumId {
        let id = SpectrumId::new(SpectrumType::Table, self.table_spectra.len());
        self.table_spectra.push(spectrum);
        id
    }

    pub fn add_constant_spectrum(&mut self, value: f32) -> SpectrumId {
        let id = SpectrumId::new(SpectrumType::Constant, self.constant_spectra.len());
        self.constant_spectra.push(ConstantSpectrum { value });
        id
    }

    pub fn add_rgb_albedo_spectrum(&mut self, rgb: Vec3) -> SpectrumId {
        let id = SpectrumId::new(SpectrumType::RgbAlbedo, self.rgb_albedo_spectra.len());
        self.rgb_albedo_spectra
            .push(RgbAlbedoSpectrum { rgb, _padding: 0 });
        id
    }

    pub fn add_rgb_ior_im_spectrum(&mut self, rgb: Vec3) -> SpectrumId {
        let id = SpectrumId::new(SpectrumType::RgbIorIm, self.rgb_ior_im_spectra.len());
        self.rgb_ior_im_spectra
            .push(RgbIorImSpectrum { rgb, _padding: 0 });
        id
    }

    pub fn add_rgb_illuminant_spectrum(&mut self, rgb: Vec3, illuminant: SpectrumId) -> SpectrumId {
        assert!(matches!(illuminant.ty(), SpectrumType::Table));
        let id = SpectrumId::new(
            SpectrumType::RgbIlluminant,
            self.rgb_illuminant_spectra.len(),
        );
        self.rgb_illuminant_spectra
            .push(RgbIlluminantSpectrum { rgb, illuminant });
        id
    }

    pub fn add_blackbody_spectrum(
        &mut self,
        temperature: f32,
        scale: f32,
        normalize: bool,
    ) -> SpectrumId {
        let normalization_factor = match normalize {
            true => self.table_spectra[1]
                .data
                .iter()
                .enumerate()
                .map(|(i, &y)| y * blackbody(i as f32 + 360.0, temperature))
                .sum::<f32>()
                .recip(),
            false => 1.0,
        };
        let id = SpectrumId::new(SpectrumType::Blackbody, self.blackbody_spectra.len());
        self.blackbody_spectra.push(BlackbodySpectrum {
            temperature,
            scale: scale * normalization_factor,
        });
        id
    }

    pub fn add_piecewise_linear_spectrum(&mut self, data: &[[f32; 2]]) -> SpectrumId {
        let id = SpectrumId::new(
            SpectrumType::PiecewiseLinear,
            self.piecewise_linear_spectra.len(),
        );
        let ptr = self.add_float_data(data.as_flattened());
        self.piecewise_linear_spectra.push(PiecewiseLinearSpectrum {
            ptr,
            entries: data.len() as u32,
        });
        id
    }
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct TableSpectrum {
    pub data: [f32; 471],
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct ConstantSpectrum {
    pub value: f32,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct RgbAlbedoSpectrum {
    pub rgb: Vec3,
    pub _padding: u32,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct RgbIlluminantSpectrum {
    pub rgb: Vec3,
    pub illuminant: SpectrumId,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct BlackbodySpectrum {
    pub temperature: f32,
    pub scale: f32,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct PiecewiseLinearSpectrum {
    pub ptr: u32,
    pub entries: u32,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct RgbIorImSpectrum {
    pub rgb: Vec3,
    pub _padding: u32,
}

fn blackbody(lambda: f32, temperature: f32) -> f32 {
    const C: f32 = 299_792_458.0;
    const H: f32 = 6.62606957e-34;
    const K_B: f32 = 1.3806488e-23;
    // smallest lambda can be is 3.6e-7, the fifth power of which is 6e-33,
    // which is still 5 orders of magnitude away from stressing the range of floats.
    let l = lambda * 1e-9;
    let l2 = l * l;
    let l5 = l2 * l2 * l;
    let e = H * C / (l * K_B * temperature);
    let radiance = 2.0 * H * C * C / (l5 * (e.exp() - 1.0));
    // Planck's law gives radiance per meter, but we use nanometers
    radiance * 1e-9
}
