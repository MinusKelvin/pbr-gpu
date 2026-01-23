#import /spectrum.wgsl

struct TextureId {
    id: u32
}

const TEXTURE_TAG_BITS: u32 = 2;
const TEXTURE_TAG_SHIFT: u32 = 32 - TEXTURE_TAG_BITS;
const TEXTURE_IDX_MASK: u32 = (1 << TEXTURE_TAG_SHIFT) - 1;
const TEXTURE_TAG_MASK: u32 = ~TEXTURE_IDX_MASK;

const TEXTURE_CONSTANT_FLOAT: u32 = 0 << TEXTURE_TAG_SHIFT;
const TEXTURE_CONSTANT_RGB: u32 = 1 << TEXTURE_TAG_SHIFT;
const TEXTURE_CONSTANT_SPECTRUM: u32 = 2 << TEXTURE_TAG_SHIFT;
const TEXTURE_IMAGE_RGB: u32 = 3 << TEXTURE_TAG_SHIFT;

@group(0) @binding(64)
var<storage> CONSTANT_FLOAT_TEXTURES: array<ConstantFloatTexture>;
@group(0) @binding(65)
var<storage> CONSTANT_RGB_TEXTURES: array<ConstantRgbTexture>;
@group(0) @binding(66)
var<storage> CONSTANT_SPECTRUM_TEXTURES: array<ConstantSpectrumTexture>;
@group(0) @binding(67)
var<storage> IMAGE_RGB_TEXTURES: array<ImageRgbTexture>;

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

fn texture_evaluate(texture_id: TextureId, uv: vec2f, wl: Wavelengths) -> vec4f {
    let idx = texture_id.id & TEXTURE_IDX_MASK;
    switch texture_id.id & TEXTURE_TAG_MASK {
        case TEXTURE_CONSTANT_FLOAT {
            return vec4f(CONSTANT_FLOAT_TEXTURES[idx].value);
        }
        case TEXTURE_CONSTANT_RGB {
            return spectrum_rgb_sample(CONSTANT_RGB_TEXTURES[idx].rgb, wl);
        }
        case TEXTURE_CONSTANT_SPECTRUM {
            return spectrum_sample(CONSTANT_SPECTRUM_TEXTURES[idx].spectrum, wl);
        }
        case TEXTURE_IMAGE_RGB {
            let tex = IMAGE_RGB_TEXTURES[idx].image_index;
            let texel = vec2u(fract(uv) * vec2f(textureDimensions(IMAGES[tex])));
            let rgb = textureLoad(IMAGES[tex], texel, 0).xyz;
            return spectrum_rgb_sample(rgb, wl);
        }
        default {
            // unreachable
            return vec4f();
        }
    }
}
