use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;

use flate2::read::GzDecoder;
use glam::{DMat3, DMat4, DVec2, DVec3, Vec3};
use lalrpop_util::{ErrorRecovery, lalrpop_mod, lexer::Token};

use crate::options::RenderOptions;
use crate::scene::{
    ImageLight, MaterialId, NodeId, PrimitiveNode, Scene, ShapeId, Sphere, TextureId, TriVertex,
};
use crate::{ProjectiveCamera, Transform};

lalrpop_mod!(grammar, "/loader/pbrt.rs");

pub fn load_pbrt_scene(path: &Path) -> (RenderOptions, Scene) {
    let mut scene = Scene::default();
    let error_texture = scene.add_constant_rgb_texture(Vec3::new(1.0, 0.0, 1.0));
    let error_material = scene.add_diffuse_material(error_texture);

    let mut builder = SceneBuilder {
        base: path.parent().unwrap().to_path_buf(),
        state: State {
            transform: DMat4::IDENTITY,
            material: error_material,
        },
        stack: vec![],
        render_options: RenderOptions::default(),
        scene,
        current_prims: vec![],
        objects: HashMap::new(),
        textures: HashMap::new(),
        materials: HashMap::new(),
        object_state: None,
        error_material,
        error_texture,
    };
    let t = Instant::now();
    builder.include(Path::new(path.file_name().unwrap()));
    eprintln!("Parse scene in {:.3?}", t.elapsed());

    let root = builder.scene.add_bvh(&builder.current_prims);
    builder.scene.root = Some(root);

    (builder.render_options, builder.scene)
}

pub struct SceneBuilder {
    base: PathBuf,
    state: State,
    stack: Vec<State>,
    error_material: MaterialId,
    error_texture: TextureId,

    render_options: RenderOptions,
    scene: Scene,

    current_prims: Vec<NodeId>,

    objects: HashMap<String, NodeId>,
    textures: HashMap<String, TextureId>,
    materials: HashMap<String, MaterialId>,

    object_state: Option<(String, Vec<NodeId>)>,
}

#[derive(Clone)]
struct State {
    transform: DMat4,
    material: MaterialId,
}

impl SceneBuilder {
    fn include(&mut self, path: &Path) {
        let content = std::fs::read_to_string(self.base.join(path)).unwrap();
        grammar::TopLevelParser::new()
            .parse(self, &content)
            .unwrap_or_else(|e| panic!("In file {}: {e}", path.display()));
    }

    fn unrecognized(&mut self, directive: &str, _err: ErrorRecovery<usize, Token<'_>, &'_ str>) {
        println!("Unrecognized directive {directive}");
    }

    fn world_begin(&mut self) {
        self.state.transform = DMat4::IDENTITY;
    }

    fn push(&mut self) {
        self.stack.push(self.state.clone());
    }

    fn pop(&mut self) {
        self.state = self.stack.pop().unwrap();
    }

    fn begin_object(&mut self, name: &str) {
        assert!(self.object_state.is_none());
        let name = name.to_owned();
        let old_prims = std::mem::take(&mut self.current_prims);
        self.object_state = Some((name, old_prims));
    }

    fn end_object(&mut self) {
        let Some((name, old_prims)) = self.object_state.take() else {
            panic!("ended object which was never started");
        };
        if self.current_prims.is_empty() {
            println!("Warning: Object {name} contains no primitives");
            return;
        }
        let obj_bvh = self.scene.add_bvh(&self.current_prims);
        self.current_prims = old_prims;
        self.objects.insert(name, obj_bvh);
    }

    fn instance_object(&mut self, name: &str) {
        let Some(&obj) = self.objects.get(name) else {
            println!("Warning: Attempt to instance object {name} which does not exist");
            return;
        };
        let transformed = self.scene.add_transform(
            Transform {
                m: self.state.transform.inverse().as_mat4(),
                m_inv: self.state.transform.as_mat4(),
            },
            obj,
        );
        self.current_prims.push(transformed);
    }

    fn identity(&mut self) {
        self.state.transform = DMat4::IDENTITY;
    }

    fn look_at(&mut self, (eye, look, up): (DVec3, DVec3, DVec3)) {
        self.state.transform *= DMat4::look_at_lh(eye, look, up);
    }

    fn rotate(&mut self, (angle, axis): (f64, DVec3)) {
        self.state.transform *= DMat4::from_axis_angle(axis, angle.to_radians());
    }

    fn translate(&mut self, offset: DVec3) {
        self.state.transform *= DMat4::from_translation(offset);
    }

    fn scale(&mut self, scale: DVec3) {
        self.state.transform *= DMat4::from_scale(scale);
    }

    fn set_transform(&mut self, mat: DMat4) {
        self.state.transform = mat;
    }

    fn apply_transform(&mut self, mat: DMat4) {
        self.state.transform *= mat;
    }

    fn camera(&mut self, kind: &str, props: Props) {
        let aspect_ratio = props
            .get_float("frameaspectratio")
            .unwrap_or(self.render_options.width as f64 / self.render_options.height as f64);
        let (ortho, mat) = match kind {
            "orhographic" => (
                true,
                DMat4::orthographic_lh(-aspect_ratio, aspect_ratio, -1.0, 1.0, 0.0, 1.0),
            ),
            "perspective" => (
                false,
                DMat4::perspective_infinite_lh(
                    props.get_float("fov").unwrap_or(90.0).to_radians(),
                    aspect_ratio,
                    0.01,
                ),
            ),
            _ => return println!("Unrecognized camera type {kind}"),
        };

        self.render_options.camera = ProjectiveCamera {
            ndc_to_camera: Transform::from_mat4_inverse(mat.as_mat4()),
            world_to_camera: Transform::from_mat4(self.state.transform.as_mat4()),
            lens_radius: props.get_float("lensradius").unwrap_or(0.0) as f32,
            focal_distance: props.get_float("focaldistance").unwrap_or(1e30) as f32,
            orthographic: ortho as u32,
            _padding: 0,
        };
    }

    fn image_texture(&mut self, name: &str, props: Props) {
        let filename = props.get_string("filename").unwrap();

        if let Some(v) = props.get_string("mapping")
            && v != "uv"
        {
            println!("Note: imagemap mapping {v} is not supported");
        }

        if let Some(v) = props.get_string("wrap")
            && v != "repeat"
        {
            println!("Note: imagemap wrapping {v} is not supported");
        }

        if props.get_float("scale").is_some() {
            println!("Note: imagemap scale is not supported");
        }

        if props.type_of("udelta").is_some() || props.type_of("vdelta").is_some() {
            println!("Note: imagemap uv-deltas not supported");
        }

        if props.type_of("uscale").is_some() || props.type_of("vscale").is_some() {
            println!("Note: imagemap uv-scale not supported");
        }

        let Some(img) = self.scene.add_image(&self.base.join(filename)) else {
            return;
        };

        let id = self.scene.add_image_texture(img);
        self.textures.insert(name.to_owned(), id);
    }

    fn constant_texture(&mut self, name: &str, props: Props) {
        let id = self.texture_property(&props, "value").unwrap();
        self.textures.insert(name.to_owned(), id);
    }

    fn scale_texture(&mut self, name: &str, props: Props) {
        let left = self
            .texture_property(&props, "tex")
            .unwrap_or_else(|| self.scene.add_constant_float_texture(1.0));
        let right = self
            .texture_property(&props, "scale")
            .unwrap_or_else(|| self.scene.add_constant_float_texture(1.0));
        let id = self.scene.add_scale_texture(left, right);
        self.textures.insert(name.to_owned(), id);
    }

    fn mix_texture(&mut self, name: &str, props: Props) {
        let tex1 = self
            .texture_property(&props, "tex1")
            .unwrap_or_else(|| self.scene.add_constant_float_texture(0.0));
        let tex2 = self
            .texture_property(&props, "tex2")
            .unwrap_or_else(|| self.scene.add_constant_float_texture(1.0));
        let amount = self
            .texture_property(&props, "amount")
            .unwrap_or_else(|| self.scene.add_constant_float_texture(0.5));
        let id = self.scene.add_mix_texture(tex1, tex2, amount);
        self.textures.insert(name.to_owned(), id);
    }

    fn checkerboard_texture(&mut self, name: &str, props: Props) {
        let even = self
            .texture_property(&props, "tex1")
            .unwrap_or_else(|| self.scene.add_constant_float_texture(1.0));
        let odd = self
            .texture_property(&props, "tex2")
            .unwrap_or_else(|| self.scene.add_constant_float_texture(0.0));
        let id = self.scene.add_checkerboard_texture(even, odd);
        self.textures.insert(name.to_owned(), id);
    }

    fn unrecognized_texture(&mut self, ty: &str) {
        println!("Unrecognized texture type {ty}");
    }

    fn texture_property(&mut self, props: &Props, name: &str) -> Option<TextureId> {
        match props.type_of(name)? {
            "texture" => Some(
                self.textures
                    .get(props.get_string(name).unwrap())
                    .copied()
                    .unwrap_or(self.error_texture),
            ),
            "rgb" => Some(
                self.scene
                    .add_constant_rgb_texture(props.get_vec3_list(name).unwrap()[0].as_vec3()),
            ),
            "float" => Some(
                self.scene
                    .add_constant_float_texture(props.get_float(name).unwrap() as f32),
            ),
            ty => {
                println!("Unrecognized texture property type {ty}");
                None
            }
        }
    }

    fn make_material(&mut self, ty: &str, props: Props) -> MaterialId {
        match ty {
            "coateddiffuse" => {
                println!("Note: coateddiffuse material will be completely diffuse");
                self.make_material("diffuse", props)
            }
            "diffuse" => {
                let texture = self
                    .texture_property(&props, "reflectance")
                    .unwrap_or_else(|| self.scene.add_constant_float_texture(0.5));
                self.scene.add_diffuse_material(texture)
            }
            "diffusetransmission" => {
                let reflectance = self
                    .texture_property(&props, "reflectance")
                    .unwrap_or_else(|| self.scene.add_constant_float_texture(0.25));
                let transmittance = self
                    .texture_property(&props, "transmittance")
                    .unwrap_or_else(|| self.scene.add_constant_float_texture(0.25));
                self.scene
                    .add_diffuse_transmit_material(reflectance, transmittance)
            }
            _ => {
                println!("Unrecognized material type {ty}");
                self.error_material
            }
        }
    }

    fn material(&mut self, ty: &str, props: Props) {
        self.state.material = self.make_material(ty, props);
    }

    fn make_named_material(&mut self, name: &str, props: Props) {
        let material = self.make_material(props.get_string("type").unwrap(), props);
        self.materials.insert(name.to_owned(), material);
    }

    fn named_material(&mut self, name: &str) {
        self.state.material = self.materials.get(name).copied().unwrap_or_else(|| {
            println!("Material {name} does not exist?");
            self.error_material
        });
    }

    fn infinite_light(&mut self, props: Props) {
        if let Some(filename) = props.get_string("filename") {
            let scale = props.get_float("scale").unwrap_or(1.0) as f32;

            let Some(image) = self.scene.add_image(&self.base.join(filename)) else {
                return;
            };
            self.scene.add_image_light(ImageLight {
                transform: Transform {
                    m: self.state.transform.as_mat4(),
                    m_inv: self.state.transform.inverse().as_mat4(),
                },
                image,
                scale,
                _padding: [0; 2],
            });
        } else {
            let color = props.get_vec3_list("L").unwrap()[0].as_vec3();
            self.scene.add_uniform_light(color);
        }
    }

    fn unrecognized_light(&mut self, ty: &str) {
        println!("Unrecognized light type {ty}");
    }

    fn sphere(&mut self, props: Props) {
        let radius = props.get_float("radius").unwrap_or(1.0);
        let z_min = props.get_float("zmin").unwrap_or(-radius);
        let z_max = props.get_float("zmax").unwrap_or(radius);

        if props.type_of("alpha").is_some() {
            println!("Node: alpha is not currently supported on spheres");
        }

        let shape_id = self.scene.add_sphere(Sphere {
            z_min: (z_min / radius) as f32,
            z_max: (z_max / radius) as f32,
            flip_normal: false as u32,
        });

        let transform = self.state.transform * DMat4::from_scale(DVec3::splat(radius));

        let primitive = self.scene.add_primitive(PrimitiveNode {
            shape: shape_id,
            material: self.state.material,
        });
        let transformed = self.scene.add_transform(
            Transform {
                m: transform.inverse().as_mat4(),
                m_inv: transform.as_mat4(),
            },
            primitive,
        );

        self.current_prims.push(transformed);
    }

    fn triangle_mesh(&mut self, props: Props) {
        let transform_dir = DMat3::from_mat4(self.state.transform);
        if transform_dir.determinant() < 0.0 {
            println!("Creating mesh with transform which swaps handedness");
        }

        if props.type_of("alpha").is_some() {
            println!("Node: alpha is not currently supported on triangles");
        }

        let indices = props
            .get_uint_list("indices")
            .unwrap_or_else(|| vec![0, 1, 2]);

        let positions = props.get_vec3_list("P").unwrap();
        let positions: Vec<_> = positions
            .into_iter()
            .map(|p| self.state.transform.transform_point3(p).as_vec3())
            .collect();

        let transform_normal = transform_dir.inverse().transpose();
        let normals = props.get_vec3_list("N").unwrap_or(vec![]);
        let normals: Vec<_> = normals
            .into_iter()
            .map(|p| transform_normal.mul_vec3(p).normalize_or_zero().as_vec3())
            .collect();

        let uvs = props.get_vec2_list("uv").unwrap_or(vec![]);

        let verts: Vec<_> = positions
            .into_iter()
            .zip(normals.into_iter().chain(std::iter::repeat(Vec3::ZERO)))
            .zip(uvs.into_iter().chain(std::iter::repeat(DVec2::ZERO)))
            .map(|((p, n), uv)| TriVertex {
                p,
                u: uv.x as f32,
                n,
                v: uv.y as f32,
            })
            .collect();

        let tris = indices
            .chunks_exact(3)
            .map(|is| is.try_into().unwrap())
            .collect::<Vec<_>>();

        let iter = self.scene.add_triangles(&verts, &tris);
        self.create_primitives(iter);
    }

    fn loop_subdivision_surface(&mut self, props: Props) {
        println!("Note: loop subdivision surface will not be subdivided.");
        self.triangle_mesh(props);
    }

    fn plymesh(&mut self, props: Props) {
        let file = props
            .get_string("filename")
            .expect("plymesh shape requires file name");
        let path = self.base.join(file);

        if props.get_string("displacement").is_some() {
            println!("Note: plymesh tessellation displacement map will not be applied.");
        }

        match path.extension().and_then(OsStr::to_str) {
            Some("gz") => {
                let iter = super::ply::load_plymesh(
                    &mut self.scene,
                    &mut BufReader::new(GzDecoder::new(File::open(path).unwrap())),
                    self.state.transform,
                );
                self.create_primitives(iter);
            }
            _ => {
                let iter = super::ply::load_plymesh(
                    &mut self.scene,
                    &mut BufReader::new(File::open(path).unwrap()),
                    self.state.transform,
                );
                self.create_primitives(iter);
            }
        };
    }

    fn unrecognized_shape(&mut self, ty: &str) {
        println!("Unrecognized shape type {ty}");
    }

    fn create_primitives(&mut self, shapes: impl Iterator<Item = ShapeId>) {
        self.current_prims.extend(shapes.map(|shape| {
            self.scene.add_primitive(PrimitiveNode {
                shape,
                material: self.state.material,
            })
        }));
    }
}

#[derive(Default)]
struct Props<'a> {
    map: HashMap<&'a str, (&'a str, Vec<Value<'a>>)>,
}

impl<'a> Props<'a> {
    fn type_of(&self, name: &str) -> Option<&'a str> {
        self.map.get(name).map(|&(ty, _)| ty)
    }

    fn get_float(&self, name: &str) -> Option<f64> {
        self.map
            .get(name)
            .filter(|&&(ty, _)| ty == "float")
            .and_then(|(_, vals)| vals.get(0))
            .and_then(Value::as_number)
    }

    fn get_uint_list(&self, name: &str) -> Option<Vec<u32>> {
        self.map
            .get(name)
            .filter(|&&(ty, _)| ty == "integer")
            .map(|(_, v)| {
                v.into_iter()
                    .map(|v| v.as_number().unwrap() as u32)
                    .collect()
            })
    }

    fn get_vec3_list(&self, name: &str) -> Option<Vec<DVec3>> {
        self.map
            .get(name)
            .filter(|&&(ty, _)| ty == "point3" || ty == "vector3" || ty == "normal" || ty == "rgb")
            .map(|(_, v)| {
                v.chunks_exact(3)
                    .map(|vs| {
                        DVec3::new(
                            vs[0].as_number().unwrap(),
                            vs[1].as_number().unwrap(),
                            vs[2].as_number().unwrap(),
                        )
                    })
                    .collect()
            })
    }

    fn get_vec2_list(&self, name: &str) -> Option<Vec<DVec2>> {
        self.map
            .get(name)
            .filter(|&&(ty, _)| ty == "point2" || ty == "vector2")
            .map(|(_, v)| {
                v.chunks_exact(2)
                    .map(|vs| DVec2::new(vs[0].as_number().unwrap(), vs[1].as_number().unwrap()))
                    .collect()
            })
    }

    fn get_string(&self, name: &str) -> Option<&'a str> {
        self.map
            .get(name)
            .filter(|&&(ty, _)| ty == "string" || ty == "texture")
            .and_then(|(_, v)| v.get(0))
            .and_then(Value::as_string)
    }
}

#[derive(Debug)]
enum Value<'a> {
    String(&'a str),
    Number(f64),
    Boolean(bool),
}

impl<'a> Value<'a> {
    fn as_string(&self) -> Option<&'a str> {
        match self {
            Value::String(v) => Some(v),
            _ => None,
        }
    }

    fn as_number(&self) -> Option<f64> {
        match self {
            Value::Number(v) => Some(*v),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Boolean(v) => Some(*v),
            _ => None,
        }
    }
}
