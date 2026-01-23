use std::io::BufRead;

use bytemuck::Zeroable;
use glam::{DMat3, DMat4};

use crate::scene::{Scene, ShapeId, TriVertex};

enum Format {
    BinaryLe,
    BinaryBe,
}

#[derive(Debug, Copy, Clone)]
enum PrimType {
    Float,
    Byte,
    Int,
}

#[derive(Debug, Copy, Clone)]
enum Type {
    Prim(PrimType),
    List(PrimType, PrimType),
}

struct Element {
    name: String,
    count: usize,
    properties: Vec<Property>,
}

enum Property {
    X,
    Y,
    Z,
    NormalX,
    NormalY,
    NormalZ,
    Indices(PrimType, PrimType),
    Unknown(Type),
}

pub fn load_plymesh<R: BufRead>(
    scene: &mut Scene,
    data: &mut R,
    transform: DMat4,
) -> impl Iterator<Item = ShapeId> + use<R> {
    let mut format = None;
    let mut elements = vec![];

    let mut line = String::new();
    loop {
        if line.is_empty() {
            if data.read_line(&mut line).unwrap() == 0 {
                break;
            }
        }

        let mut words = line.split_whitespace();

        match words.next().unwrap() {
            "ply" | "comment" => {}
            "end_header" => break,
            "format" => {
                format = Some(match words.next().unwrap() {
                    "binary_little_endian" => {
                        assert_eq!(
                            words.next().unwrap(),
                            "1.0",
                            "only version 1.0 of binary_little_endian is supported"
                        );
                        Format::BinaryLe
                    }
                    "binary_big_endian" => {
                        assert_eq!(
                            words.next().unwrap(),
                            "1.0",
                            "only version 1.0 of binary_big_endian is supported"
                        );
                        Format::BinaryBe
                    }
                    s => panic!("Unrecognized ply format: {s}"),
                })
            }
            "element" => {
                let name = words.next().unwrap().to_owned();
                let count = words.next().unwrap().parse().unwrap();
                let mut properties = vec![];

                loop {
                    line.clear();
                    if data.read_line(&mut line).unwrap() == 0 {
                        break;
                    }

                    let mut words = line.split_whitespace();
                    if words.next().unwrap() != "property" {
                        break;
                    }

                    let ty = match words.next().unwrap() {
                        "list" => Type::List(
                            prim_type(words.next().unwrap()),
                            prim_type(words.next().unwrap()),
                        ),
                        ty => Type::Prim(prim_type(ty)),
                    };

                    let prop = match (ty, words.next().unwrap()) {
                        (Type::Prim(PrimType::Float), "x") => Property::X,
                        (Type::Prim(PrimType::Float), "y") => Property::Y,
                        (Type::Prim(PrimType::Float), "z") => Property::Z,
                        (Type::Prim(PrimType::Float), "nx") => Property::NormalX,
                        (Type::Prim(PrimType::Float), "ny") => Property::NormalY,
                        (Type::Prim(PrimType::Float), "nz") => Property::NormalZ,
                        (Type::List(count, elem), "vertex_indices") => {
                            Property::Indices(count, elem)
                        }
                        (ty, s) => {
                            println!("Unrecognized property {ty:?} {s}");
                            Property::Unknown(ty)
                        }
                    };

                    properties.push(prop);
                }

                elements.push(Element {
                    name,
                    count,
                    properties,
                });

                continue;
            }
            s => panic!("Unrecognized ply directive: {s}"),
        }

        line.clear();
    }

    let mut format: Box<dyn FormatReader> = match format.unwrap() {
        Format::BinaryLe => Box::new(BinaryLeFormat(data)),
        Format::BinaryBe => Box::new(BinaryBeFormat(data)),
    };

    let mut vertices = vec![];
    let mut indices = vec![];

    for element in elements {
        match &*element.name {
            "vertex" => {
                let transform_dir = DMat3::from_mat4(transform);
                if transform_dir.determinant() < 0.0 {
                    println!("Creating mesh with transform which swaps handedness");
                }
                let transform_normal = transform_dir.inverse().transpose();
                for _ in 0..element.count {
                    let mut data = TriVertex::zeroed();
                    for prop in &element.properties {
                        match prop {
                            Property::X => data.p.x = format.read_float(),
                            Property::Y => data.p.y = format.read_float(),
                            Property::Z => data.p.z = format.read_float(),
                            Property::NormalX => data.n.x = format.read_float(),
                            Property::NormalY => data.n.y = format.read_float(),
                            Property::NormalZ => data.n.z = format.read_float(),
                            _ => format.skip(prop.ty()),
                        }
                    }
                    vertices.push(TriVertex {
                        p: transform.transform_point3(data.p.as_dvec3()).as_vec3(),
                        _padding0: 0,
                        n: transform_normal
                            .mul_vec3(data.n.as_dvec3())
                            .normalize_or_zero()
                            .as_vec3(),
                        _padding1: 0,
                    });
                }
            }
            "face" => {
                for _ in 0..element.count {
                    for prop in &element.properties {
                        match prop {
                            &Property::Indices(count_ty, elem_ty) => {
                                let count = format.read_int(count_ty);
                                let idx: Vec<_> =
                                    (0..count).map(|_| format.read_int(elem_ty)).collect();
                                for i in 2..count as usize {
                                    indices.push([idx[0], idx[i - 1], idx[i]]);
                                }
                            }
                            _ => format.skip(prop.ty()),
                        }
                    }
                }
            }
            s => {
                println!("Unrecognized ply element {s}");
                for _ in 0..element.count {
                    for prop in &element.properties {
                        format.skip(prop.ty());
                    }
                }
            }
        }
    }

    scene.add_triangles(&vertices, &indices)
}

fn prim_type(name: &str) -> PrimType {
    match name {
        "float" => PrimType::Float,
        "uint8" | "uchar" => PrimType::Byte,
        "int" | "uint" => PrimType::Int,
        _ => panic!("Unrecognized ply type: {name}"),
    }
}

impl Property {
    fn ty(&self) -> Type {
        match self {
            Property::X
            | Property::Y
            | Property::Z
            | Property::NormalX
            | Property::NormalY
            | Property::NormalZ => Type::Prim(PrimType::Float),
            Property::Indices(count, elem) => Type::List(*count, *elem),
            Property::Unknown(ty) => *ty,
        }
    }
}

trait FormatReader {
    fn read_float(&mut self) -> f32;
    fn read_u8(&mut self) -> u8;
    fn read_u32(&mut self) -> u32;

    fn read_int(&mut self, ty: PrimType) -> u32 {
        match ty {
            PrimType::Float => self.read_float() as u32,
            PrimType::Byte => self.read_u8() as u32,
            PrimType::Int => self.read_u32(),
        }
    }

    fn skip(&mut self, ty: Type) {
        match ty {
            Type::Prim(PrimType::Byte) => {
                self.read_u8();
            }
            Type::Prim(PrimType::Int) => {
                self.read_u32();
            }
            Type::Prim(PrimType::Float) => {
                self.read_float();
            }
            Type::List(count_ty, elem_ty) => {
                let count = self.read_int(count_ty);
                for _ in 0..count {
                    self.skip(Type::Prim(elem_ty));
                }
            }
        }
    }
}

struct BinaryLeFormat<R>(R);

impl<R: BufRead> FormatReader for BinaryLeFormat<R> {
    fn read_float(&mut self) -> f32 {
        let mut buf = [0; 4];
        self.0.read_exact(&mut buf).unwrap();
        f32::from_le_bytes(buf)
    }

    fn read_u8(&mut self) -> u8 {
        let mut buf = [0; 1];
        self.0.read_exact(&mut buf).unwrap();
        buf[0]
    }

    fn read_u32(&mut self) -> u32 {
        let mut buf = [0; 4];
        self.0.read_exact(&mut buf).unwrap();
        u32::from_le_bytes(buf)
    }
}

struct BinaryBeFormat<R>(R);

impl<R: BufRead> FormatReader for BinaryBeFormat<R> {
    fn read_float(&mut self) -> f32 {
        let mut buf = [0; 4];
        self.0.read_exact(&mut buf).unwrap();
        f32::from_be_bytes(buf)
    }

    fn read_u8(&mut self) -> u8 {
        let mut buf = [0; 1];
        self.0.read_exact(&mut buf).unwrap();
        buf[0]
    }

    fn read_u32(&mut self) -> u32 {
        let mut buf = [0; 4];
        self.0.read_exact(&mut buf).unwrap();
        u32::from_be_bytes(buf)
    }
}
