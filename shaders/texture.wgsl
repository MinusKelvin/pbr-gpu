#import /spectrum.wgsl

struct TextureId {
    id: u32
}

const TEXTURE_TAG_BITS: u32 = 3;
const TEXTURE_TAG_SHIFT: u32 = 32 - TEXTURE_TAG_BITS;
const TEXTURE_IDX_MASK: u32 = (1 << TEXTURE_TAG_SHIFT) - 1;
const TEXTURE_TAG_MASK: u32 = ~TEXTURE_IDX_MASK;

const TEXTURE_CONSTANT: u32 = 0 << TEXTURE_TAG_SHIFT;
const TEXTURE_IMAGE_RGB: u32 = 3 << TEXTURE_TAG_SHIFT;
const TEXTURE_SCALE: u32 = 4 << TEXTURE_TAG_SHIFT;
const TEXTURE_MIX: u32 = 5 << TEXTURE_TAG_SHIFT;
const TEXTURE_CHECKERBOARD: u32 = 6 << TEXTURE_TAG_SHIFT;

@group(0) @binding(64)
var<storage> CONSTANT_TEXTURES: array<ConstantTexture>;
@group(0) @binding(67)
var<storage> IMAGE_RGB_TEXTURES: array<ImageRgbTexture>;
@group(0) @binding(68)
var IMAGES: binding_array<texture_2d<f32>>;
@group(0) @binding(69)
var<storage> SCALE_TEXTURES: array<ScaleTexture>;
@group(0) @binding(70)
var<storage> MIX_TEXTURES: array<MixTexture>;
@group(0) @binding(71)
var<storage> CHECKERBOARD_TEXTURES: array<CheckerboardTexture>;

struct ConstantTexture {
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

struct CheckerboardTexture {
    even: TextureId,
    odd: TextureId,
}

fn texture_evaluate(texture_id: TextureId, uv: vec2f, wl: Wavelengths) -> vec4f {
    var tex_stack: array<TextureId, 8>;
    var data: array<vec4f, 8>;

    var tex_i = 1;
    var data_i = 0;
    tex_stack[0] = texture_id;

    while tex_i > 0 {
        tex_i--;
        let idx = tex_stack[tex_i].id & TEXTURE_IDX_MASK;
        let tag = tex_stack[tex_i].id & TEXTURE_TAG_MASK;
        if idx != TEXTURE_IDX_MASK {
            // pre-eval step; push data and/or required texture evaluations
            switch tag {
                case TEXTURE_CONSTANT {
                    data[data_i] = spectrum_sample(CONSTANT_TEXTURES[idx].spectrum, wl);
                    data_i++;
                }
                case TEXTURE_IMAGE_RGB {
                    let tex = IMAGE_RGB_TEXTURES[idx].image_index;
                    let mapped = fract(uv);
                    let texel = vec2f(mapped.x, (1 - EPSILON/2) - mapped.y)
                        * vec2f(textureDimensions(IMAGES[tex]));
                    let rgb = textureLoad(IMAGES[tex], vec2u(texel), 0).xyz;
                    data[data_i] = spectrum_rgb_albedo_sample(RgbAlbedoSpectrum(rgb), wl);
                    data_i++;
                }
                case TEXTURE_SCALE {
                    tex_stack[tex_i].id |= TEXTURE_IDX_MASK;
                    tex_i++;

                    tex_stack[tex_i] = SCALE_TEXTURES[idx].right;
                    tex_i++;
                    tex_stack[tex_i] = SCALE_TEXTURES[idx].left;
                    tex_i++;
                }
                case TEXTURE_MIX {
                    tex_stack[tex_i].id |= TEXTURE_IDX_MASK;
                    tex_i++;

                    tex_stack[tex_i] = MIX_TEXTURES[idx].amount;
                    tex_i++;
                    tex_stack[tex_i] = MIX_TEXTURES[idx].tex2;
                    tex_i++;
                    tex_stack[tex_i] = MIX_TEXTURES[idx].tex1;
                    tex_i++;
                }
                case TEXTURE_CHECKERBOARD {
                    let odd = (i32(floor(uv.x)) + i32(floor(uv.y))) % 2 != 0;
                    if odd {
                        tex_stack[tex_i] = CHECKERBOARD_TEXTURES[idx].odd;
                    } else {
                        tex_stack[tex_i] = CHECKERBOARD_TEXTURES[idx].even;
                    }
                    tex_i++;
                }
                default {
                    // unreachable
                    return vec4f();
                }
            }
        } else {
            // post-eval step; data from other textures is on the stack
            switch tag {
                case TEXTURE_SCALE {
                    data_i--;
                    let left = data[data_i];
                    data_i--;
                    let right = data[data_i];
                    data[data_i] = left * right;
                    data_i++;
                }
                case TEXTURE_MIX {
                    data_i--;
                    let tex1 = data[data_i];
                    data_i--;
                    let tex2 = data[data_i];
                    data_i--;
                    let amount = data[data_i];
                    data[data_i] = mix(tex1, tex2, amount);
                    data_i++;
                }
                default {
                    // unreachable
                    return vec4f();
                }
            }
        }
    }

    if data_i != 1 {
        return spectrum_rgb_albedo_sample(RgbAlbedoSpectrum(vec3f(1, 0, 1)), wl);
    }

    return data[0];
}
