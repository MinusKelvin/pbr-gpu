use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;

use flate2::read::GzDecoder;
use glam::{DMat3, DMat4, DVec2, DVec3, Vec2, Vec3};
use lalrpop_util::{ErrorRecovery, lalrpop_mod, lexer::Token};

use crate::options::RenderOptions;
use crate::scene::{
     LightId, MaterialId, NodeId, PrimitiveNode, Scene, ShapeId, SpectrumId, Sphere,
    TextureId, TriVertex, UvMappingParams,
};
use crate::spectrum::SpectrumData;
use crate::{ProjectiveCamera, Transform};

lalrpop_mod!(grammar, "/loader/pbrt.rs");

pub fn load_pbrt_scene(spectrum_data: &SpectrumData, path: &Path) -> (RenderOptions, Scene) {
    let mut scene = Scene::new(spectrum_data);
    let spectrum = scene.add_rgb_albedo_spectrum(Vec3::new(1.0, 0.0, 1.0));
    let error_texture = scene.add_constant_texture(spectrum);
    let error_material = scene.add_diffuse_material(error_texture);

    let mut builder = SceneBuilder {
        base: path.parent().unwrap().to_path_buf(),
        state: State {
            transform: DMat4::IDENTITY,
            material: error_material,
            area_light: None,
        },
        stack: vec![],
        render_options: RenderOptions::default(),
        scene,
        current_prims: vec![],
        lights: vec![],
        objects: HashMap::new(),
        textures: HashMap::new(),
        materials: HashMap::new(),
        object_state: None,
        error_material,
        error_texture,
    };
    let t = Instant::now();
    builder.include(Path::new(path.file_name().unwrap()));

    let root = builder.scene.add_bvh(&builder.current_prims);
    builder.scene.root = Some(root);

    let root_ls = builder.scene.add_power_light_sampler(&builder.lights);
    builder.scene.root_ls = Some(root_ls);

    eprintln!("Build scene in {:.3?}", t.elapsed());

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
    lights: Vec<LightId>,

    objects: HashMap<String, NodeId>,
    textures: HashMap<String, TextureId>,
    materials: HashMap<String, MaterialId>,

    object_state: Option<(String, Vec<NodeId>)>,
}

#[derive(Clone)]
struct State {
    transform: DMat4,
    material: MaterialId,
    area_light: Option<(SpectrumId, bool)>,
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

    fn image_texture(&mut self, name: &str, kind: &str, props: Props) {
        let filename = props.get_string("filename").unwrap();

        let is_float = match kind {
            "spectrum" => false,
            "float" => true,
            _ => {
                println!("Unrecognized texture kind {kind}");
                return;
            }
        };

        let uv_map = self.uv_mapping(&props);

        let Some(img) = self.scene.add_image(&self.base.join(filename), is_float) else {
            return;
        };

        let id = match is_float {
            true => self.scene.add_float_image_texture(img, uv_map),
            false => self.scene.add_rgb_image_texture(img, uv_map),
        };
        self.textures.insert(name.to_owned(), id);
    }

    fn constant_texture(&mut self, name: &str, props: Props) {
        let id = self.texture_property(&props, "value").unwrap();
        self.textures.insert(name.to_owned(), id);
    }

    fn scale_texture(&mut self, name: &str, props: Props) {
        let left = self.texture_property(&props, "tex").unwrap_or_else(|| {
            let spec = self.scene.add_constant_spectrum(1.0);
            self.scene.add_constant_texture(spec)
        });
        let right = self.texture_property(&props, "scale").unwrap_or_else(|| {
            let spec = self.scene.add_constant_spectrum(1.0);
            self.scene.add_constant_texture(spec)
        });
        let id = self.scene.add_scale_texture(left, right);
        self.textures.insert(name.to_owned(), id);
    }

    fn mix_texture(&mut self, name: &str, props: Props) {
        let tex1 = self.texture_property(&props, "tex1").unwrap_or_else(|| {
            let spec = self.scene.add_constant_spectrum(0.0);
            self.scene.add_constant_texture(spec)
        });
        let tex2 = self.texture_property(&props, "tex2").unwrap_or_else(|| {
            let spec = self.scene.add_constant_spectrum(1.0);
            self.scene.add_constant_texture(spec)
        });
        let amount = self.texture_property(&props, "amount").unwrap_or_else(|| {
            let spec = self.scene.add_constant_spectrum(0.5);
            self.scene.add_constant_texture(spec)
        });
        let id = self.scene.add_mix_texture(tex1, tex2, amount);
        self.textures.insert(name.to_owned(), id);
    }

    fn checkerboard_texture(&mut self, name: &str, props: Props) {
        let even = self.texture_property(&props, "tex1").unwrap_or_else(|| {
            let spec = self.scene.add_constant_spectrum(1.0);
            self.scene.add_constant_texture(spec)
        });
        let odd = self.texture_property(&props, "tex2").unwrap_or_else(|| {
            let spec = self.scene.add_constant_spectrum(0.0);
            self.scene.add_constant_texture(spec)
        });
        let uv_map = self.uv_mapping(&props);
        let id = self.scene.add_checkerboard_texture(even, odd, uv_map);
        self.textures.insert(name.to_owned(), id);
    }

    fn unrecognized_texture(&mut self, ty: &str) {
        println!("Unrecognized texture type {ty}");
    }

    fn uv_mapping(&self, props: &Props) -> UvMappingParams {
        if let Some(mapping) = props.get_string("mapping")
            && mapping != "uv"
        {
            println!("Warning: Unsupported texture mapping mode {mapping}");
        }
        let mut uv_map = UvMappingParams {
            scale: Vec2::ONE,
            delta: Vec2::ZERO,
        };
        if let Some(u_scale) = props.get_float("uscale") {
            uv_map.scale.x = u_scale as f32;
        }
        if let Some(v_scale) = props.get_float("vscale") {
            uv_map.scale.y = v_scale as f32;
        }
        if let Some(u_delta) = props.get_float("udelta") {
            uv_map.delta.x = u_delta as f32;
        }
        if let Some(v_delta) = props.get_float("vdelta") {
            uv_map.delta.y = v_delta as f32;
        }
        uv_map
    }

    fn spectrum_property(
        &mut self,
        props: &Props,
        name: &str,
        scale: f32,
        illum: bool,
    ) -> Option<SpectrumId> {
        match props.type_of(name)? {
            "rgb" if illum => Some(self.scene.add_rgb_illuminant_spectrum(
                props.get_vec3_list(name).unwrap()[0].as_vec3(),
                SpectrumId::D65,
            )),
            "rgb" => Some(
                self.scene
                    .add_rgb_albedo_spectrum(props.get_vec3_list(name).unwrap()[0].as_vec3()),
            ),
            "float" => Some(
                self.scene
                    .add_constant_spectrum(props.get_float(name).unwrap() as f32),
            ),
            "blackbody" => Some(self.scene.add_blackbody_spectrum(
                props.get_float(name).unwrap() as f32,
                scale,
                true,
            )),
            "spectrum" => {
                if let Some(&spectrum) = props
                    .get_string(name)
                    .and_then(|name| self.scene.named_spectra.get(name))
                {
                    Some(spectrum)
                } else if let Some(file) = props.get_string(name) {
                    let path = self.base.join(file);
                    let content = std::fs::read_to_string(&path)
                        .unwrap_or_else(|e| panic!("{e}: {}", path.display()));
                    let data: Vec<_> = content
                        .lines()
                        .filter(|l| !l.contains('#'))
                        .filter_map(|l| l.split_once(char::is_whitespace))
                        .map(|(l, v)| [l.trim().parse().unwrap(), v.trim().parse().unwrap()])
                        .collect();
                    Some(self.scene.add_piecewise_linear_spectrum(&data))
                } else if let Some(data) = props.get_float_list(name) {
                    let data: Vec<_> = data
                        .chunks_exact(2)
                        .map(|a| [a[0] as f32, a[1] as f32])
                        .collect();
                    Some(self.scene.add_piecewise_linear_spectrum(&data))
                } else {
                    println!("Could not interpret spectrum");
                    None
                }
            }
            ty => {
                println!("Unrecognized spectrum property type {ty}");
                None
            }
        }
    }

    fn texture_property(&mut self, props: &Props, name: &str) -> Option<TextureId> {
        match props.type_of(name)? {
            "texture" => Some(
                self.textures
                    .get(props.get_string(name).unwrap())
                    .copied()
                    .unwrap_or(self.error_texture),
            ),
            _ => self
                .spectrum_property(props, name, 1.0, false)
                .map(|spectrum| self.scene.add_constant_texture(spectrum)),
        }
    }

    fn make_material(&mut self, ty: &str, props: Props) -> MaterialId {
        match ty {
            "coateddiffuse" => self.make_material("metallicworkflow", props),
            "coatedconductor" => {
                println!("Note: coatedconductor material will be regular conductor");
                let mut props = props;
                if let Some(data) = props.map.remove("conductor.eta") {
                    props.map.insert("eta", data);
                }
                if let Some(data) = props.map.remove("conductor.k") {
                    props.map.insert("k", data);
                }
                if let Some(data) = props.map.remove("conductor.roughness") {
                    props.map.insert("roughness", data);
                }
                self.make_material("conductor", props)
            }
            "diffuse" => {
                let texture = self
                    .texture_property(&props, "reflectance")
                    .unwrap_or_else(|| {
                        let spec = self.scene.add_constant_spectrum(0.5);
                        self.scene.add_constant_texture(spec)
                    });
                self.scene.add_diffuse_material(texture)
            }
            "diffusetransmission" => {
                let reflectance =
                    self.texture_property(&props, "reflectance")
                        .unwrap_or_else(|| {
                            let spec = self.scene.add_constant_spectrum(0.25);
                            self.scene.add_constant_texture(spec)
                        });
                let transmittance = self
                    .texture_property(&props, "transmittance")
                    .unwrap_or_else(|| {
                        let spec = self.scene.add_constant_spectrum(0.25);
                        self.scene.add_constant_texture(spec)
                    });
                let scale = self.texture_property(&props, "scale").unwrap_or_else(|| {
                    let spec = self.scene.add_constant_spectrum(1.0);
                    self.scene.add_constant_texture(spec)
                });

                self.scene
                    .add_diffuse_transmit_material(reflectance, transmittance, scale)
            }
            "conductor" => {
                let refl = props.get_vec3_list("reflectance");

                let (ior_re, ior_im) = match refl {
                    Some(refl) => (
                        self.scene.add_constant_spectrum(1.0),
                        self.scene.add_rgb_ior_im_spectrum(refl[0].as_vec3()),
                    ),
                    None => (
                        self.spectrum_property(&props, "eta", 1.0, false)
                            .unwrap_or(self.scene.named_spectra["metal-Cu-eta"]),
                        self.spectrum_property(&props, "k", 1.0, false)
                            .unwrap_or(self.scene.named_spectra["metal-Cu-k"]),
                    ),
                };

                let u_roughness = self.texture_property(&props, "uroughness");
                let v_roughness = self.texture_property(&props, "vroughness");
                let (u_roughness, v_roughness) = u_roughness
                    .zip(v_roughness)
                    .inspect(|_| println!("Note: anisotropic roughness currently not supported"))
                    .unwrap_or_else(|| {
                        let roughness =
                            self.texture_property(&props, "roughness")
                                .unwrap_or_else(|| {
                                    let spec = self.scene.add_constant_spectrum(0.0);
                                    self.scene.add_constant_texture(spec)
                                });
                        (roughness, roughness)
                    });

                self.scene
                    .add_conductor_material(ior_re, ior_im, u_roughness, v_roughness)
            }
            "dielectric" => {
                let ior = self
                    .spectrum_property(&props, "eta", 1.0, false)
                    .unwrap_or_else(|| self.scene.add_constant_spectrum(1.5));

                let u_roughness = self.texture_property(&props, "uroughness");
                let v_roughness = self.texture_property(&props, "vroughness");
                let (u_roughness, v_roughness) = u_roughness
                    .zip(v_roughness)
                    .inspect(|_| println!("Note: anisotropic roughness currently not supported"))
                    .unwrap_or_else(|| {
                        let roughness =
                            self.texture_property(&props, "roughness")
                                .unwrap_or_else(|| {
                                    let spec = self.scene.add_constant_spectrum(0.0);
                                    self.scene.add_constant_texture(spec)
                                });
                        (roughness, roughness)
                    });

                self.scene
                    .add_dielectric_material(ior, u_roughness, v_roughness)
            }
            "thindielectric" => {
                let ior = self
                    .spectrum_property(&props, "eta", 1.0, false)
                    .unwrap_or_else(|| self.scene.add_constant_spectrum(1.5));

                self.scene.add_thin_dielectric_material(ior)
            }
            "metallicworkflow" => {
                let base_color =
                    self.texture_property(&props, "reflectance")
                        .unwrap_or_else(|| {
                            let spec = self.scene.add_constant_spectrum(0.5);
                            self.scene.add_constant_texture(spec)
                        });

                let metallic = self
                    .texture_property(&props, "metallic")
                    .unwrap_or_else(|| {
                        let spec = self.scene.add_constant_spectrum(0.0);
                        self.scene.add_constant_texture(spec)
                    });

                let u_roughness = self.texture_property(&props, "uroughness");
                let v_roughness = self.texture_property(&props, "vroughness");
                let (u_roughness, v_roughness) = u_roughness
                    .zip(v_roughness)
                    .inspect(|_| println!("Note: anisotropic roughness currently not supported"))
                    .unwrap_or_else(|| {
                        let roughness =
                            self.texture_property(&props, "roughness")
                                .unwrap_or_else(|| {
                                    let spec = self.scene.add_constant_spectrum(0.0);
                                    self.scene.add_constant_texture(spec)
                                });
                        (roughness, roughness)
                    });

                self.scene.add_metallic_workflow_material(
                    base_color,
                    metallic,
                    u_roughness,
                    v_roughness,
                )
            }
            "mix" => {
                let materials = props.get_string_list("materials").unwrap();
                let m1 = materials[0];
                let m2 = materials[1];

                let m1 = self.materials.get(m1).copied().unwrap_or_else(|| {
                    println!("Material {m1} does not exist?");
                    self.error_material
                });
                let m2 = self.materials.get(m2).copied().unwrap_or_else(|| {
                    println!("Material {m2} does not exist?");
                    self.error_material
                });

                let amount = self.texture_property(&props, "amount").unwrap_or_else(|| {
                    let spec = self.scene.add_constant_spectrum(0.5);
                    self.scene.add_constant_texture(spec)
                });

                self.scene.add_mix_material(m1, m2, amount)
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
        let scale = props.get_float("scale").unwrap_or(1.0) as f32;
        if let Some(filename) = props.get_string("filename") {
            let Some(image) = self.scene.add_image(&self.base.join(filename), false) else {
                return;
            };
            let light = self
                .scene
                .add_image_light(self.state.transform, image, scale);
            self.lights.push(light);
        } else if let Some(spectrum) = self.spectrum_property(&props, "L", scale, true) {
            let light = self.scene.add_uniform_light(spectrum);
            self.lights.push(light);
        } else {
            println!("Infinite light specifies neither image nor spectrum?");
        }
    }

    fn unrecognized_light(&mut self, ty: &str) {
        println!("Unrecognized light type {ty}");
    }

    fn diffuse_area_light(&mut self, props: Props) {
        let scale = props.get_float("scale").unwrap_or(1.0) as f32;
        let two_sided = props.get_bool("twosided").unwrap_or(false);
        self.state.area_light = self
            .spectrum_property(&props, "L", scale, true)
            .map(|l| (l, two_sided));
    }

    fn unrecognized_area_light(&mut self, ty: &str) {
        println!("Unrecognized area light type {ty}");
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

        let one = self.scene.add_constant_spectrum(1.0);
        let one = self.scene.add_constant_texture(one);

        let light = match self.state.area_light {
            Some((spectrum, two_sided)) => {
                println!("Note: light sampling spheres is currently not supported");
                self.scene
                    .add_area_light(shape_id, spectrum, two_sided, one)
            }
            None => LightId::ZERO,
        };

        let primitive = self.scene.add_primitive(PrimitiveNode {
            shape: shape_id,
            material: self.state.material,
            light,
            alpha: one,
        });
        let transformed = self.scene.add_transform(
            Transform {
                m: transform.inverse().as_mat4(),
                m_inv: transform.as_mat4(),
            },
            primitive,
        );

        if light != LightId::ZERO {
            self.scene.set_area_light_transform(light, transformed);
            self.lights.push(light);
        }

        self.current_prims.push(transformed);
    }

    fn triangle_mesh(&mut self, props: Props) {
        let transform_dir = DMat3::from_mat4(self.state.transform);
        if transform_dir.determinant() < 0.0 {
            println!("Creating mesh with transform which swaps handedness");
        }

        let alpha = self.texture_property(&props, "alpha").unwrap_or_else(|| {
            let one = self.scene.add_constant_spectrum(1.0);
            self.scene.add_constant_texture(one)
        });

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
        self.create_primitives(alpha, iter);
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

        let alpha = self.texture_property(&props, "alpha").unwrap_or_else(|| {
            let one = self.scene.add_constant_spectrum(1.0);
            self.scene.add_constant_texture(one)
        });

        match path.extension().and_then(OsStr::to_str) {
            Some("gz") => {
                let iter = super::ply::load_plymesh(
                    &mut self.scene,
                    &mut BufReader::new(GzDecoder::new(File::open(path).unwrap())),
                    self.state.transform,
                );
                self.create_primitives(alpha, iter);
            }
            _ => {
                let iter = super::ply::load_plymesh(
                    &mut self.scene,
                    &mut BufReader::new(File::open(path).unwrap()),
                    self.state.transform,
                );
                self.create_primitives(alpha, iter);
            }
        };
    }

    fn unrecognized_shape(&mut self, ty: &str) {
        println!("Unrecognized shape type {ty}");
    }

    fn create_primitives(&mut self, alpha: TextureId, shapes: impl Iterator<Item = ShapeId>) {
        self.current_prims.extend(shapes.map(|shape| {
            let light = match self.state.area_light {
                Some((rgb, two_sided)) => self.scene.add_area_light(shape, rgb, two_sided, alpha),
                None => LightId::ZERO,
            };
            if light != LightId::ZERO {
                self.lights.push(light);
            }
            self.scene.add_primitive(PrimitiveNode {
                shape,
                material: self.state.material,
                light,
                alpha,
            })
        }));
    }
}

#[derive(Default)]
struct Props<'a> {
    map: HashMap<&'a str, (&'a str, Vec<Value<'a>>)>,
    used: RefCell<HashSet<&'a str>>,
    ctx: &'a str,
    domain: &'a str,
}

impl<'a> Props<'a> {
    fn with_ctx(mut self, domain: &'a str, ctx: &'a str) -> Self {
        Props {
            map: std::mem::take(&mut self.map),
            used: std::mem::take(&mut self.used),
            ctx,
            domain,
        }
    }

    fn lookup(&self, name: &str) -> Option<&(&'a str, Vec<Value<'a>>)> {
        let (k, v) = self.map.get_key_value(name)?;
        self.used.borrow_mut().insert(k);
        Some(v)
    }

    fn type_of(&self, name: &str) -> Option<&'a str> {
        self.lookup(name).map(|&(ty, _)| ty)
    }

    fn get_float(&self, name: &str) -> Option<f64> {
        self.lookup(name)
            .filter(|&&(ty, _)| ty == "float" || ty == "blackbody")
            .and_then(|(_, vals)| vals.get(0))
            .and_then(Value::as_number)
    }

    fn get_float_list(&self, name: &str) -> Option<Vec<f64>> {
        self.lookup(name)
            .filter(|&&(ty, _)| ty == "float" || ty == "spectrum")
            .and_then(|(_, vals)| vals.into_iter().map(|v| v.as_number()).collect())
    }

    fn get_uint_list(&self, name: &str) -> Option<Vec<u32>> {
        self.lookup(name)
            .filter(|&&(ty, _)| ty == "integer")
            .map(|(_, v)| {
                v.into_iter()
                    .map(|v| v.as_number().unwrap() as u32)
                    .collect()
            })
    }

    fn get_vec3_list(&self, name: &str) -> Option<Vec<DVec3>> {
        self.lookup(name)
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
        self.lookup(name)
            .filter(|&&(ty, _)| ty == "point2" || ty == "vector2")
            .map(|(_, v)| {
                v.chunks_exact(2)
                    .map(|vs| DVec2::new(vs[0].as_number().unwrap(), vs[1].as_number().unwrap()))
                    .collect()
            })
    }

    fn get_string(&self, name: &str) -> Option<&'a str> {
        self.lookup(name)
            .filter(|&&(ty, _)| ty == "string" || ty == "texture" || ty == "spectrum")
            .and_then(|(_, v)| v.get(0))
            .and_then(Value::as_string)
    }

    fn get_string_list(&self, name: &str) -> Option<Vec<&'a str>> {
        self.lookup(name)
            .filter(|&&(ty, _)| ty == "string")
            .and_then(|(_, vals)| vals.into_iter().map(|v| v.as_string()).collect())
    }

    fn get_bool(&self, name: &str) -> Option<bool> {
        self.lookup(name)
            .filter(|&&(ty, _)| ty == "bool")
            .and_then(|(_, vals)| vals.get(0))
            .and_then(Value::as_bool)
    }
}

impl Drop for Props<'_> {
    #[track_caller]
    fn drop(&mut self) {
        let ctx = self.ctx;
        let domain = self.domain;
        if ctx.is_empty() {
            return;
        }
        let used = self.used.borrow();
        for &name in self.map.keys() {
            if !used.contains(name) {
                println!("Warning: Unknown property {name} in {ctx} {domain}");
            }
        }
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
