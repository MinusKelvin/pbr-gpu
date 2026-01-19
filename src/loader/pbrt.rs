use std::collections::HashMap;
use std::path::{Path, PathBuf};

use glam::{DMat4, DVec3, Vec3};
use lalrpop_util::{ErrorRecovery, lalrpop_mod, lexer::Token};

use crate::options::RenderOptions;
use crate::scene::{PrimitiveNode, Scene, Sphere};
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
    };
    builder.include(Path::new(path.file_name().unwrap()));

    let root = builder.scene.add_bvh(&builder.current_prims);
    builder.scene.root = root;

    (builder.render_options, builder.scene)
}

pub struct SceneBuilder {
    base: PathBuf,
    state: State,
    stack: Vec<State>,

    render_options: RenderOptions,
    scene: Scene,

    current_prims: Vec<u32>,
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
            .unwrap();
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

    fn identity(&mut self) {
        self.state.transform = DMat4::IDENTITY;
    }

    fn look_at(&mut self, (eye, look, up): (DVec3, DVec3, DVec3)) {
        self.state.transform *= DMat4::look_at_lh(eye, look, up);
    }

    fn rotate(&mut self, (angle, axis): (f64, DVec3)) {
        self.state.transform *= DMat4::from_axis_angle(axis, angle);
    }

    fn translate(&mut self, offset: DVec3) {
        self.state.transform *= DMat4::from_translation(offset);
    }

    fn scale(&mut self, scale: DVec3) {
        self.state.transform *= DMat4::from_scale(scale);
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

        let primitive = self.scene.add_primitive(PrimitiveNode { shape: shape_id });

        self.current_prims.push(primitive);
    }

    fn triangle_mesh(&mut self, props: Props) {
        let indices = props
            .get_uint_list("indices")
            .unwrap_or_else(|| vec![0, 1, 2]);
        let positions = props.get_vec3_list("P").unwrap();
        let positions: Vec<_> = positions
            .into_iter()
            .map(|p| self.state.transform.transform_point3(p).as_vec3())
            .collect();

        let base_index = self.scene.add_triangle_vertices(&positions);
        let tris = indices
            .chunks_exact(3)
            .map(|is| [is[0] + base_index, is[1] + base_index, is[2] + base_index])
            .collect::<Vec<_>>();
        let base_shape_id = self.scene.add_triangles(&tris);

        for i in 0..tris.len() {
            let primitive = self.scene.add_primitive(PrimitiveNode {
                shape: base_shape_id + i as u32,
            });
            self.current_prims.push(primitive);
        }
    }

    fn loop_subdivision_surface(&mut self, props: Props) {
        eprintln!("Note: loop subdivision surface will not be subdivided.");
        self.triangle_mesh(props);
    }

    fn unrecognized_shape(&mut self, ty: &str) {
        eprintln!("Unrecognized shape type {ty}");
    }
}

#[derive(Default)]
struct Props<'a> {
    map: HashMap<&'a str, (&'a str, Vec<Value<'a>>)>,
}

impl<'a> Props<'a> {
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
