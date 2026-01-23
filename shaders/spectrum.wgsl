const WAVELENGTH_MIN: f32 = 360;
const WAVELENGTH_MAX: f32 = 830;

const SPECTRUM_CIE_X = SpectrumId(0);
const SPECTRUM_CIE_Y = SpectrumId(1);
const SPECTRUM_CIE_Z = SpectrumId(2);
const SPECTRUM_D65_1NIT = SpectrumId(3);

@group(1) @binding(2)
var<storage> SPECTRA: array<DenseSpectrum>;

@group(1) @binding(3)
var RGB_TO_COEFF: texture_3d<f32>;

struct SpectrumId {
    id: u32
}

struct Wavelengths {
    l: vec4f,
}

struct DenseSpectrum {
    data: array<f32, u32(WAVELENGTH_MAX - WAVELENGTH_MIN + 1)>,
}

fn spectrum_sample(spectrum: SpectrumId, wl: Wavelengths) -> vec4f {
    let i = vec4u(wl.l - WAVELENGTH_MIN);
    return vec4(
        mix(SPECTRA[spectrum.id].data[i.x], SPECTRA[spectrum.id].data[i.x + 1u], fract(wl.l.x)),
        mix(SPECTRA[spectrum.id].data[i.y], SPECTRA[spectrum.id].data[i.y + 1u], fract(wl.l.y)),
        mix(SPECTRA[spectrum.id].data[i.z], SPECTRA[spectrum.id].data[i.z + 1u], fract(wl.l.z)),
        mix(SPECTRA[spectrum.id].data[i.w], SPECTRA[spectrum.id].data[i.w + 1u], fract(wl.l.w)),
    );
}

fn spectrum_rgb_sample(rgb: vec3f, wl: Wavelengths) -> vec4f {
    let dim = textureDimensions(RGB_TO_COEFF);
    let texel = vec3u(rgb * vec3f(dim));
    let coeffs = textureLoad(RGB_TO_COEFF, min(texel, dim - 1), 0);

    let l = (wl.l - WAVELENGTH_MIN) / (WAVELENGTH_MAX - WAVELENGTH_MIN);
    let poly = coeffs.x * l * l + coeffs.y * l + coeffs.z;

    return 0.5 + poly / (2 * sqrt(1 + poly * poly));
}
