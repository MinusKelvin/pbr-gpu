use std::fmt::Write;

use bytemuck::NoUninit;
use glam::{Vec2, Vec3, Vec4};

use crate::scene::{Scene, SpectrumId};

pub trait Texture {
    fn emit(&self, builder: &mut TextureEvaluatorBuilder) -> Variable;

    fn is_spectrum(&self) -> bool;

    fn box_clone(&self) -> Box<dyn Texture>;
}

impl Clone for Box<dyn Texture> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

impl Scene {
    pub fn float_texture_evaluator(&mut self, texture: &dyn Texture) -> FloatTextureId {
        assert!(!texture.is_spectrum());

        let mut builder = TextureEvaluatorBuilder {
            num_vars: 0,
            eval_code: String::new(),
        };
        let result = texture.emit(&mut builder);
        _ = writeln!(&mut builder.eval_code, "return {result};");

        let next_id = self.float_code.len() as u32;
        let id = *self
            .float_code
            .entry(builder.eval_code)
            .or_insert_with_key(|code| {
                _ = writeln!(&mut self.float_texture_match, "case {next_id} {{{code}}}");
                next_id
            });

        FloatTextureId(id)
    }

    pub fn spectrum_texture_evaluator(&mut self, texture: &dyn Texture) -> SpectrumTextureId {
        assert!(texture.is_spectrum());

        let mut builder = TextureEvaluatorBuilder {
            num_vars: 0,
            eval_code: String::new(),
        };
        let result = texture.emit(&mut builder);
        _ = writeln!(&mut builder.eval_code, "return {result};");

        let next_id = self.spectrum_code.len() as u32;
        let id = *self
            .spectrum_code
            .entry(builder.eval_code)
            .or_insert_with_key(|code| {
                _ = writeln!(
                    &mut self.spectrum_texture_match,
                    "case {next_id} {{{code}}}"
                );
                next_id
            });

        SpectrumTextureId(id)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, NoUninit)]
#[repr(C)]
pub struct FloatTextureId(u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, NoUninit)]
#[repr(C)]
pub struct SpectrumTextureId(u32);

impl FloatTextureId {
    pub fn raw(self) -> u32 {
        self.0
    }
}

pub struct TextureEvaluatorBuilder {
    num_vars: usize,
    eval_code: String,
}

#[derive(Copy, Clone, Debug)]
pub enum Parameter {
    Float(f32),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
    U32(u32),
}

#[derive(Copy, Clone, Debug)]
pub enum Variable {
    Let(usize),
    Param(Parameter),
}

impl TextureEvaluatorBuilder {
    fn parameter(&mut self, p: Parameter) -> Variable {
        Variable::Param(p)
    }

    fn variable(&mut self) -> Variable {
        let id = self.num_vars;
        self.num_vars += 1;
        Variable::Let(id)
    }

    fn emit(&mut self, code: impl AsRef<str>) {
        self.eval_code.push_str(code.as_ref());
        self.eval_code.push('\n');
    }
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Variable::Let(id) => write!(f, "v{id}"),
            Variable::Param(Parameter::Float(v)) => write!(f, "{v}"),
            Variable::Param(Parameter::Vec2(v)) => write!(f, "vec2f({}, {})", v.x, v.y),
            Variable::Param(Parameter::Vec3(v)) => write!(f, "vec3f({}, {}, {})", v.x, v.y, v.z),
            Variable::Param(Parameter::Vec4(v)) => {
                write!(f, "vec4f({}, {}, {}, {})", v.x, v.y, v.z, v.w)
            }
            Variable::Param(Parameter::U32(v)) => write!(f, "{v}u"),
        }
    }
}

#[derive(Copy, Clone)]
pub struct ConstantFloatTexture {
    pub value: f32,
}

impl Texture for ConstantFloatTexture {
    fn emit(&self, builder: &mut TextureEvaluatorBuilder) -> Variable {
        builder.parameter(Parameter::Float(self.value))
    }

    fn is_spectrum(&self) -> bool {
        false
    }

    fn box_clone(&self) -> Box<dyn Texture> {
        Box::new(self.clone())
    }
}

#[derive(Copy, Clone)]
pub struct ConstantSpectrumTexture {
    pub spectrum: SpectrumId,
}

impl Texture for ConstantSpectrumTexture {
    fn emit(&self, builder: &mut TextureEvaluatorBuilder) -> Variable {
        let result = builder.variable();
        let param = builder.parameter(Parameter::U32(self.spectrum.raw()));
        builder.emit(format!(
            "let {result} = spectrum_sample(SpectrumId({param}), wavelengths);"
        ));
        result
    }

    fn is_spectrum(&self) -> bool {
        true
    }

    fn box_clone(&self) -> Box<dyn Texture> {
        Box::new(self.clone())
    }
}

#[derive(Copy, Clone)]
pub struct UvMappingParams {
    pub scale: Vec2,
    pub delta: Vec2,
}

#[derive(Copy, Clone)]
pub struct ImageRgbTexture {
    pub image: u32,
    pub scale: f32,
    pub invert: u32,
    pub uv_map: UvMappingParams,
}

impl Texture for ImageRgbTexture {
    fn emit(&self, builder: &mut TextureEvaluatorBuilder) -> Variable {
        let image = builder.parameter(Parameter::U32(self.image));
        let scale = builder.parameter(Parameter::Float(self.scale));
        let invert = builder.parameter(Parameter::U32(self.invert));
        let uv_scale = builder.parameter(Parameter::Vec2(self.uv_map.scale));
        let uv_delta = builder.parameter(Parameter::Vec2(self.uv_map.delta));

        let mapped = builder.variable();
        builder.emit(format!("let {mapped} = uv * {uv_scale} + {uv_delta};"));

        let rgb = builder.variable();
        builder.emit(format!(
            "var {rgb} = textureSampleLevel(
            IMAGES[{image}], LINEAR_FILTER_WRAP, vec2({mapped}.x, 1 - {mapped}.y), 0
        ).xyz * {scale};"
        ));

        builder.emit(format!(
            "if {invert} != 0 {{ {rgb} = max(vec3f(), vec3f(1) - {rgb}); }}"
        ));

        let result = builder.variable();
        builder.emit(format!(
            "let {result} = spectrum_rgb_albedo_sample(RgbAlbedoSpectrum({rgb}), wavelengths);"
        ));

        result
    }

    fn is_spectrum(&self) -> bool {
        true
    }

    fn box_clone(&self) -> Box<dyn Texture> {
        Box::new(self.clone())
    }
}

#[derive(Copy, Clone)]
pub struct ImageFloatTexture {
    pub image: u32,
    pub scale: f32,
    pub invert: u32,
    pub uv_map: UvMappingParams,
}

impl Texture for ImageFloatTexture {
    fn emit(&self, builder: &mut TextureEvaluatorBuilder) -> Variable {
        let image = builder.parameter(Parameter::U32(self.image));
        let scale = builder.parameter(Parameter::Float(self.scale));
        let invert = builder.parameter(Parameter::U32(self.invert));
        let uv_scale = builder.parameter(Parameter::Vec2(self.uv_map.scale));
        let uv_delta = builder.parameter(Parameter::Vec2(self.uv_map.delta));

        let mapped = builder.variable();
        builder.emit(format!("let {mapped} = uv * {uv_scale} + {uv_delta};"));

        let result = builder.variable();
        builder.emit(format!(
            "var {result} = textureSampleLevel(
            IMAGES[{image}], LINEAR_FILTER_WRAP, vec2({mapped}.x, 1 - {mapped}.y), 0
        ).x * {scale};"
        ));

        builder.emit(format!("if {invert} != 0 {{ {result} = 1 - {result}; }}"));

        result
    }

    fn is_spectrum(&self) -> bool {
        false
    }

    fn box_clone(&self) -> Box<dyn Texture> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct ConductorReflTexture {
    pub color: Box<dyn Texture>,
}

impl Texture for ConductorReflTexture {
    fn emit(&self, builder: &mut TextureEvaluatorBuilder) -> Variable {
        let color = self.color.emit(builder);
        let result = builder.variable();
        builder.emit(format!(
            "let {result} = 2 * sqrt({color}) / sqrt(1 - {color});"
        ));

        result
    }

    fn is_spectrum(&self) -> bool {
        true
    }

    fn box_clone(&self) -> Box<dyn Texture> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct ScaleTexture {
    pub left: Box<dyn Texture>,
    pub right: Box<dyn Texture>,
}

impl Texture for ScaleTexture {
    fn emit(&self, builder: &mut TextureEvaluatorBuilder) -> Variable {
        let left = self.left.emit(builder);
        let right = self.right.emit(builder);
        let result = builder.variable();
        builder.emit(format!("let {result} = {left} * {right};"));
        result
    }

    fn is_spectrum(&self) -> bool {
        self.left.is_spectrum()
    }

    fn box_clone(&self) -> Box<dyn Texture> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct MixTexture {
    pub tex1: Box<dyn Texture>,
    pub tex2: Box<dyn Texture>,
    pub amount: Box<dyn Texture>,
}

impl Texture for MixTexture {
    fn emit(&self, builder: &mut TextureEvaluatorBuilder) -> Variable {
        let tex1 = self.tex1.emit(builder);
        let tex2 = self.tex2.emit(builder);
        let amount = self.amount.emit(builder);
        let result = builder.variable();
        builder.emit(format!("let {result} = mix({tex1}, {tex2}, {amount});"));
        result
    }

    fn is_spectrum(&self) -> bool {
        self.tex1.is_spectrum()
    }

    fn box_clone(&self) -> Box<dyn Texture> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct CheckerboardTexture {
    pub even: Box<dyn Texture>,
    pub odd: Box<dyn Texture>,
    pub uv_map: UvMappingParams,
}

impl Texture for CheckerboardTexture {
    fn emit(&self, builder: &mut TextureEvaluatorBuilder) -> Variable {
        let uv_scale = builder.parameter(Parameter::Vec2(self.uv_map.scale));
        let uv_delta = builder.parameter(Parameter::Vec2(self.uv_map.delta));

        let result = builder.variable();
        builder.emit(format!("var {result}: {};", match self.is_spectrum() {
            true => "vec4f",
            false => "f32",
        }));

        let mapped = builder.variable();
        builder.emit(format!("let {mapped} = vec2i(floor(uv * {uv_scale} + {uv_delta}));"));
        builder.emit(format!("if ({mapped}.x + {mapped}.y) % 2 != 0 {{"));

        let odd = self.odd.emit(builder);
        builder.emit(format!("{result} = {odd};"));

        builder.emit("} else {");

        let even = self.even.emit(builder);
        builder.emit(format!("{result} = {even};"));

        builder.emit("}");

        result
    }

    fn is_spectrum(&self) -> bool {
        self.even.is_spectrum()
    }

    fn box_clone(&self) -> Box<dyn Texture> {
        Box::new(self.clone())
    }
}
