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

fn float_texture_derivative(id: FloatTextureId, uv: vec2f) -> vec2f {
    let delta = 0.00005;

    let v_p = float_texture_evaluate(id, uv);
    let v_pu = float_texture_evaluate(id, uv + vec2(delta, 0));
    let v_pv = float_texture_evaluate(id, uv + vec2(0, delta));

    return (vec2(v_pu, v_pv) - vec2(v_p)) / delta;
}
