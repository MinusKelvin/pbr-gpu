#import /spectrum.wgsl

struct TextureId {
    id: u32
}

const TEXTURE_TAG_BITS: u32 = 3;
const TEXTURE_TAG_SHIFT: u32 = 32 - TEXTURE_TAG_BITS;
const TEXTURE_IDX_MASK: u32 = (1 << TEXTURE_TAG_SHIFT) - 1;
const TEXTURE_TAG_MASK: u32 = ~TEXTURE_IDX_MASK;

const TEXTURE_CONSTANT_FLOAT: u32 = 0 << TEXTURE_TAG_SHIFT;
const TEXTURE_CONSTANT_RGB: u32 = 1 << TEXTURE_TAG_SHIFT;
const TEXTURE_CONSTANT_SPECTRUM: u32 = 2 << TEXTURE_TAG_SHIFT;
const TEXTURE_IMAGE_RGB: u32 = 3 << TEXTURE_TAG_SHIFT;
const TEXTURE_SCALE: u32 = 4 << TEXTURE_TAG_SHIFT;
const TEXTURE_MIX: u32 = 5 << TEXTURE_TAG_SHIFT;

@group(0) @binding(64)
var<storage> CONSTANT_FLOAT_TEXTURES: array<ConstantFloatTexture>;
@group(0) @binding(65)
var<storage> CONSTANT_RGB_TEXTURES: array<ConstantRgbTexture>;
@group(0) @binding(66)
var<storage> CONSTANT_SPECTRUM_TEXTURES: array<ConstantSpectrumTexture>;
@group(0) @binding(67)
var<storage> IMAGE_RGB_TEXTURES: array<ImageRgbTexture>;
@group(0) @binding(69)
var<storage> SCALE_TEXTURES: array<ScaleTexture>;
@group(0) @binding(70)
var<storage> MIX_TEXTURES: array<MixTexture>;

@group(0) @binding(68)
var IMAGES: binding_array<texture_2d<f32>>;

struct ConstantFloatTexture {
    value: f32
}

struct ConstantRgbTexture {
    rgb: vec3f
}

struct ConstantSpectrumTexture {
    spectrum: SpectrumId
}

struct ImageRgbTexture {
    image_index: u32,
}

struct ScaleTexture {
    left: TextureId,
    right: TextureId,
}

struct MixTexture {
    tex1: TextureId,
    tex2: TextureId,
    amount: TextureId,
}

fn texture_evaluate(texture_id: TextureId, uv: vec2f, wl: Wavelengths) -> vec4f {
    var pre_eval: array<TextureId, 8>;
    var post_eval: array<TextureId, 8>;
    var data: array<vec4f, 8>;

    var pre_i = 1;
    var data_i = 0;
    var post_i = 0;
    pre_eval[0] = texture_id;

    while pre_i > 0 {
        pre_i -= 1;
        let idx = pre_eval[pre_i].id & TEXTURE_IDX_MASK;
        let tag = pre_eval[pre_i].id & TEXTURE_TAG_MASK;
        switch tag {
            case TEXTURE_CONSTANT_FLOAT {
                data[data_i] = vec4f(CONSTANT_FLOAT_TEXTURES[idx].value);
                data_i += 1;
            }
            case TEXTURE_CONSTANT_RGB {
                data[data_i] = spectrum_rgb_sample(CONSTANT_RGB_TEXTURES[idx].rgb, wl);
                data_i += 1;
            }
            case TEXTURE_CONSTANT_SPECTRUM {
                data[data_i] = spectrum_sample(CONSTANT_SPECTRUM_TEXTURES[idx].spectrum, wl);
                data_i += 1;
            }
            case TEXTURE_IMAGE_RGB {
                let tex = IMAGE_RGB_TEXTURES[idx].image_index;
                let texel = vec2u(fract(uv) * vec2f(textureDimensions(IMAGES[tex])));
                let rgb = textureLoad(IMAGES[tex], texel, 0).xyz;
                data[data_i] = spectrum_rgb_sample(rgb, wl);
                data_i += 1;
            }
            case TEXTURE_SCALE {
                post_eval[post_i] = pre_eval[pre_i];
                post_i += 1;
                pre_eval[pre_i] = SCALE_TEXTURES[idx].right;
                pre_i += 1;
                pre_eval[pre_i] = SCALE_TEXTURES[idx].left;
                pre_i += 1;
            }
            case TEXTURE_MIX {
                post_eval[post_i] = pre_eval[pre_i];
                post_i += 1;
                pre_eval[pre_i] = MIX_TEXTURES[idx].amount;
                pre_i += 1;
                pre_eval[pre_i] = MIX_TEXTURES[idx].tex2;
                pre_i += 1;
                pre_eval[pre_i] = MIX_TEXTURES[idx].tex1;
                pre_i += 1;
            }
            default {
                // unreachable
                return vec4f();
            }
        }
    }

    while post_i > 0 {
        post_i -= 1;
        let idx = post_eval[post_i].id & TEXTURE_IDX_MASK;
        let tag = post_eval[post_i].id & TEXTURE_TAG_MASK;
        switch tag {
            case TEXTURE_SCALE {
                data_i -= 1;
                let left = data[data_i];
                data_i -= 1;
                let right = data[data_i];
                data[data_i] = left * right;
                data_i += 1;
            }
            case TEXTURE_MIX {
                data_i -= 1;
                let tex1 = data[data_i];
                data_i -= 1;
                let tex2 = data[data_i];
                data_i -= 1;
                let amount = data[data_i];
                data[data_i] = mix(tex1, tex2, amount);
                data_i += 1;
            }
            default {
                // unreachable
                return vec4f();
            }
        }
    }

    return data[0];
}
