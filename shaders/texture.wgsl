#import /spectrum.wgsl

struct SpectrumTextureId {
    id: u32
}

struct FloatTextureId {
    id: u32
}

@group(0) @binding(68)
var IMAGES: binding_array<texture_2d<f32>>;

@group(1) @binding(25)
var LINEAR_FILTER_WRAP: sampler;

// generated functions
// fn spectrum_texture_evaluate(id: SpectrumTextureId, uv: vec2f, wavelengths: Wavelengths) -> vec4f;
// fn float_texture_evaluate(id: FloatTextureId, uv: vec2f) -> f32;
