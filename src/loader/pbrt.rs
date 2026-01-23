use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;

use flate2::read::GzDecoder;
use glam::{DMat3, DMat4, DVec3, Vec3};
use lalrpop_util::{ErrorRecovery, lalrpop_mod, lexer::Token};

use crate::options::RenderOptions;
use crate::scene::{NodeId, PrimitiveNode, Scene, ShapeId, Sphere, TriVertex};
use crate::{ProjectiveCamera, Transform};

lalrpop_mod!(grammar, "/loader/pbrt.rs");

pub fn load_pbrt_scene(path: &Path) -> (RenderOptions, Scene) {
    let mut builder = SceneBuilder {
        base: path.parent().unwrap().to_path_buf(),
        state: State {
            transform: DMat4::IDENTITY,
        },
        stack: vec![],
        render_options: RenderOptions::default(),
        scene: Scene::default(),
        current_prims: vec![],
        objects: HashMap::new(),
        object_state: None,
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

    render_options: RenderOptions,
    scene: Scene,

    current_prims: Vec<NodeId>,

    objects: HashMap<String, NodeId>,

    object_state: Option<(String, Vec<NodeId>)>,
}

#[derive(Clone)]
struct State {
    transform: DMat4,
}

impl SceneBuilder {
    fn include(&mut self, path: &Path) {
        let content = std::fs::read_to_string(self.base.join(path)).unwrap();
        grammar::TopLevelParser::new()
            .parse(self, &content)
            .unwrap_or_else(|e| panic!("In file {}: {e}", path.display()));
    }

    fn unrecognized(&mut self, directive: &str, _err: ErrorRecovery<usize, Token<'_>, &'_ str>) {
        eprintln!("Unrecognized directive {directive}");
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
            eprintln!("Warning: Object {name} contains no primitives");
            return;
        }
        let obj_bvh = self.scene.add_bvh(&self.current_prims);
        self.current_prims = old_prims;
        self.objects.insert(name, obj_bvh);
    }

    fn instance_object(&mut self, name: &str) {
        let Some(&obj) = self.objects.get(name) else {
            eprintln!("Warning: Attempt to instance object {name} which does not exist");
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
            _ => return eprintln!("Unrecognized camera type {kind}"),
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

    fn sphere(&mut self, props: Props) {
        let radius = props.get_float("radius").unwrap_or(1.0);
        let z_min = props.get_float("zmin").unwrap_or(-radius);
        let z_max = props.get_float("zmax").unwrap_or(radius);

        let shape_id = self.scene.add_sphere(Sphere {
            z_min: (z_min / radius) as f32,
            z_max: (z_max / radius) as f32,
            flip_normal: false as u32,
        });

        let transform = self.state.transform * DMat4::from_scale(DVec3::splat(radius));

        let primitive = self.scene.add_primitive(PrimitiveNode { shape: shape_id });
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
            eprintln!("Creating mesh with transform which swaps handedness");
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

        let verts: Vec<_> = positions
            .into_iter()
            .zip(normals.into_iter().chain(std::iter::repeat(Vec3::ZERO)))
            .map(|(p, n)| TriVertex {
                p,
                _padding0: 0,
                n,
                _padding1: 0,
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
        eprintln!("Note: loop subdivision surface will not be subdivided.");
        self.triangle_mesh(props);
    }

    fn plymesh(&mut self, props: Props) {
        let file = props
            .get_string("filename")
            .expect("plymesh shape requires file name");
        let path = self.base.join(file);

        if props.get_string("displacement").is_some() {
            eprintln!("Note: plymesh tessellation displacement map will not be applied.");
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
        eprintln!("Unrecognized shape type {ty}");
    }

    fn create_primitives(&mut self, shapes: impl Iterator<Item = ShapeId>) {
        self.current_prims
            .extend(shapes.map(|shape| self.scene.add_primitive(PrimitiveNode { shape })));
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
            .filter(|&&(ty, _)| ty == "point3" || ty == "vector3" || ty == "normal")
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
