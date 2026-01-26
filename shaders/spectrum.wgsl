const WAVELENGTH_MIN: f32 = 360;
const WAVELENGTH_MAX: f32 = 831;

const SPECTRUM_CIE_X = SpectrumId(0);
const SPECTRUM_CIE_Y = SpectrumId(1);
const SPECTRUM_CIE_Z = SpectrumId(2);
const SPECTRUM_D65_1NIT = SpectrumId(3);

@group(0) @binding(160)
var<storage> TABLE_SPECTRA: array<TableSpectrum>;
@group(0) @binding(161)
var<storage> CONSTANT_SPECTRA: array<ConstantSpectrum>;
@group(0) @binding(162)
var<storage> RGB_ALBEDO_SPECTRA: array<RgbAlbedoSpectrum>;
@group(0) @binding(163)
var<storage> RGB_ILLUMINANT_SPECTRA: array<RgbIlluminantSpectrum>;
@group(0) @binding(164)
var<storage> BLACKBODY_SPECTRA: array<BlackbodySpectrum>;

const SPECTRUM_TAG_BITS: u32 = 3;
const SPECTRUM_TAG_SHIFT: u32 = 32 - SPECTRUM_TAG_BITS;
const SPECTRUM_IDX_MASK: u32 = (1 << SPECTRUM_TAG_SHIFT) - 1;
const SPECTRUM_TAG_MASK: u32 = ~SPECTRUM_IDX_MASK;

const SPECTRUM_TABLE: u32 = 0 << SPECTRUM_TAG_SHIFT;
const SPECTRUM_CONSTANT: u32 = 1 << SPECTRUM_TAG_SHIFT;
const SPECTRUM_RGB_ALBEDO: u32 = 2 << SPECTRUM_TAG_SHIFT;
const SPECTRUM_RGB_ILLUMINANT: u32 = 3 << SPECTRUM_TAG_SHIFT;
const SPECTRUM_BLACKBODY: u32 = 4 << SPECTRUM_TAG_SHIFT;

@group(1) @binding(32)
var RGB_TO_COEFF: texture_3d<f32>;

struct SpectrumId {
    id: u32
}

struct Wavelengths {
    l: vec4f,
}

struct TableSpectrum {
    data: array<f32, u32(WAVELENGTH_MAX - WAVELENGTH_MIN)>,
}

struct ConstantSpectrum {
    value: f32,
}

struct RgbAlbedoSpectrum {
    rgb: vec3f,
}

struct RgbIlluminantSpectrum {
    rgb: vec3f,
    illum: SpectrumId,
}

struct BlackbodySpectrum {
    temperature: f32,
    scale: f32,
}

fn spectrum_sample(spectrum: SpectrumId, wl: Wavelengths) -> vec4f {
    let idx = spectrum.id & SPECTRUM_IDX_MASK;
    switch spectrum.id & SPECTRUM_TAG_MASK {
        case SPECTRUM_TABLE {
            return spectrum_table_sample(idx, wl);
        }
        case SPECTRUM_CONSTANT {
            return vec4f(CONSTANT_SPECTRA[idx].value);
        }
        case SPECTRUM_RGB_ALBEDO {
            return spectrum_rgb_albedo_sample(RGB_ALBEDO_SPECTRA[idx], wl);
        }
        case SPECTRUM_RGB_ILLUMINANT {
            return spectrum_rgb_illuminant_sample(RGB_ILLUMINANT_SPECTRA[idx], wl);
        }
        case SPECTRUM_BLACKBODY {
            return spectrum_blackbody_sample(BLACKBODY_SPECTRA[idx], wl);
        }
        default {
            // unreachable
            return vec4f();
        }
    }
}

fn spectrum_table_sample(idx: u32, wl: Wavelengths) -> vec4f {
    let i = vec4u(wl.l - WAVELENGTH_MIN);
    return vec4(
        TABLE_SPECTRA[idx].data[i.x],
        TABLE_SPECTRA[idx].data[i.y],
        TABLE_SPECTRA[idx].data[i.z],
        TABLE_SPECTRA[idx].data[i.w],
    );
}

fn spectrum_rgb_albedo_sample(spectrum: RgbAlbedoSpectrum, wl: Wavelengths) -> vec4f {
    let dim = textureDimensions(RGB_TO_COEFF);
    let texel = vec3u(spectrum.rgb * vec3f(dim));
    let coeffs = textureLoad(RGB_TO_COEFF, min(texel, dim - 1), 0);

    let l = (wl.l - WAVELENGTH_MIN) / (WAVELENGTH_MAX - WAVELENGTH_MIN);
    let poly = coeffs.x * l * l + coeffs.y * l + coeffs.z;

    return 0.5 + poly / (2 * sqrt(1 + poly * poly));
}

fn spectrum_rgb_illuminant_sample(spectrum: RgbIlluminantSpectrum, wl: Wavelengths) -> vec4f {
    let m = max(spectrum.rgb.x, max(spectrum.rgb.y, spectrum.rgb.z));
    let scale = m * 2;
    if scale == 0 {
        return vec4f();
    } else {
        return spectrum_rgb_albedo_sample(RgbAlbedoSpectrum(spectrum.rgb / scale), wl)
            * spectrum_table_sample(spectrum.illum.id, wl)
            * scale;
    }
}

fn spectrum_blackbody_sample(spectrum: BlackbodySpectrum, wl: Wavelengths) -> vec4f {
    return spectrum.scale * vec4f(
        blackbody(wl.l.x, spectrum.temperature),
        blackbody(wl.l.y, spectrum.temperature),
        blackbody(wl.l.z, spectrum.temperature),
        blackbody(wl.l.w, spectrum.temperature),
    );
}
