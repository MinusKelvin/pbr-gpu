const WAVELENGTH_MIN: f32 = 360;
const WAVELENGTH_MAX: f32 = 830;

const SPECTRUM_CIE_X = SpectrumId(0);
const SPECTRUM_CIE_Y = SpectrumId(1);
const SPECTRUM_CIE_Z = SpectrumId(2);
const SPECTRUM_D65_1NIT = SpectrumId(3);

@group(1) @binding(2)
var<storage> spectra: array<DenseSpectrum>;

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
        mix(spectra[spectrum.id].data[i.x], spectra[spectrum.id].data[i.x + 1u], fract(wl.l.x)),
        mix(spectra[spectrum.id].data[i.y], spectra[spectrum.id].data[i.y + 1u], fract(wl.l.y)),
        mix(spectra[spectrum.id].data[i.z], spectra[spectrum.id].data[i.z + 1u], fract(wl.l.z)),
        mix(spectra[spectrum.id].data[i.w], spectra[spectrum.id].data[i.w + 1u], fract(wl.l.w)),
    );
}
