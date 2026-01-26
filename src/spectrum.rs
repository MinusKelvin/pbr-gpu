use std::error::Error;
use std::io::Read;
use std::path::Path;

use glam::{DMat3, DVec3, FloatExt, Mat3, USizeVec3, Vec3};
use ordered_float::OrderedFloat;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::scene::TableSpectrum;

pub const RGB_COEFF_N: u32 = 64;
const RGB_COEFF_SIZE: usize = (RGB_COEFF_N as usize).pow(3);

pub struct SpectrumData {
    pub cie_x: Box<TableSpectrum>,
    pub cie_y: Box<TableSpectrum>,
    pub cie_z: Box<TableSpectrum>,
    pub d65: Box<TableSpectrum>,

    pub rgb_coeffs: Vec<[f32; 4]>,
}

pub fn load_data() -> Result<SpectrumData, Box<dyn Error>> {
    // scale XYZ such that 1 W of 555nm light is 683.002 nits
    let [cie_x, cie_y, cie_z] = load_spectrum("spectrum/CIE_xyz_1931_2deg.csv", 683.002)?;
    let y_int = cie_y.data.iter().sum::<f32>();
    // scale D65 to 1 nit brightness.
    // standard D65 is scaled such that int(D65*Y) = 100 when Y is scaled to have integral 1
    let [d65] = load_spectrum("spectrum/CIE_std_illum_D65.csv", 1.0 / (y_int * 100.0))?;

    let rgb_cache_path = ".rgbcache";
    let rgb_coeffs = std::fs::File::open(rgb_cache_path)
        .and_then(|mut file| {
            let mut data = vec![[0.0; 4]; RGB_COEFF_SIZE];
            file.read_exact(bytemuck::cast_slice_mut(&mut data))?;
            Ok(data)
        })
        .unwrap_or_else(|e| {
            println!("Could not load RGB coefficients ({e}), will recompute");
            let data = compute_rgb_coeffs(&cie_x, &cie_y, &cie_z, &d65);
            if let Err(e) = std::fs::write(rgb_cache_path, bytemuck::cast_slice(&data)) {
                println!("Failed to save RGB coefficients ({e})");
            }
            data
        });

    Ok(SpectrumData {
        cie_x,
        cie_y,
        cie_z,
        d65,
        rgb_coeffs,
    })
}

fn load_spectrum<const N: usize>(
    path: &str,
    scale: f32,
) -> Result<[Box<TableSpectrum>; N], Box<dyn Error>> {
    let mut piecewise = annotate_path(path.as_ref(), load_csv::<N>)?;
    piecewise
        .iter_mut()
        .for_each(|(_, v)| *v = v.map(|v| v * scale));
    Ok(piecewise_to_densely_sampled(piecewise))
}

fn annotate_path<T>(
    path: &Path,
    f: impl FnOnce(&Path) -> Result<T, Box<dyn Error>>,
) -> Result<T, Box<dyn Error>> {
    Ok(f(path).map_err(|e| std::io::Error::other(format!("{e} (in {})", path.display())))?)
}

fn load_csv<const N: usize>(path: &Path) -> Result<Vec<(f32, [f32; N])>, Box<dyn Error>> {
    std::fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let (wl, values) = line
                .split_once(',')
                .ok_or_else(|| std::io::Error::other("row lacks values"))?;
            let wl = wl.trim().parse()?;
            let values: Vec<_> = values
                .split(',')
                .map(str::parse)
                .collect::<Result<_, _>>()?;
            Ok((
                wl,
                values
                    .try_into()
                    .map_err(|_| std::io::Error::other("row has incorrect number of values"))?,
            ))
        })
        .collect()
}

fn piecewise_to_densely_sampled<const N: usize>(
    f: Vec<(f32, [f32; N])>,
) -> [Box<TableSpectrum>; N] {
    (0..N)
        .map(|j| {
            let data = (360..=830)
                .map(|wl| {
                    match f
                        .binary_search_by_key(&OrderedFloat(wl as f32), |&(wl, _)| OrderedFloat(wl))
                    {
                        Ok(i) => f[i].1[j],
                        Err(i) => {
                            let t = (wl as f32 - f[i - 1].0) / (f[i].0 - f[i - 1].0);
                            f[i - 1].1[j].lerp(f[i].1[j], t)
                        }
                    }
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            Box::new(TableSpectrum { data })
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn compute_rgb_coeffs(
    x: &TableSpectrum,
    y: &TableSpectrum,
    z: &TableSpectrum,
    white: &TableSpectrum,
) -> Vec<[f32; 4]> {
    const SRGB_TO_XYZ_T: Mat3 = Mat3::from_cols_array_2d(&[
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ]);
    let xyz_to_srgb = SRGB_TO_XYZ_T.transpose().inverse();

    let srgb_matching = x
        .data
        .iter()
        .zip(&y.data)
        .zip(&z.data)
        .map(|((&x, &y), &z)| xyz_to_srgb * Vec3::new(x, y, z))
        .collect::<Vec<_>>();

    let mut data = vec![];
    (0..RGB_COEFF_SIZE)
        .into_par_iter()
        .map(|i| {
            let r = i % RGB_COEFF_N as usize;
            let g = i / RGB_COEFF_N as usize % RGB_COEFF_N as usize;
            let b = i / RGB_COEFF_N as usize / RGB_COEFF_N as usize;
            compute_rgb_coefficient(
                &srgb_matching,
                white,
                (USizeVec3::new(r, g, b).as_dvec3() + 0.5) / 64.0,
            )
        })
        .collect_into_vec(&mut data);
    data
}

fn compute_rgb_coefficient(matching: &[Vec3], white: &TableSpectrum, rgb: DVec3) -> [f32; 4] {
    let mut coeffs = DVec3::ZERO;
    let mut best = coeffs;
    let mut best_err = f64::INFINITY;

    for i in 0.. {
        let color = compute_color(matching, white, coeffs);
        let err = rgb.distance_squared(color);

        if err < best_err {
            // println!("{i} {coeffs} {color} {rgb} {err}");
            best_err = err;
            best = coeffs;
        }

        if err < 1e-6 || i > 15 {
            if best_err > 1e-3 {
                println!(
                    "{rgb} approximated as {}",
                    compute_color(matching, white, best)
                );
            }
            break;
        }

        let delta = 0.00001;
        let r_low = DMat3::from_cols_array_2d(
            &DVec3::AXES
                .map(|axis| rgb - compute_color(matching, white, coeffs - axis * delta))
                .map(|v| v.to_array()),
        );
        let r_high = DMat3::from_cols_array_2d(
            &DVec3::AXES
                .map(|axis| rgb - compute_color(matching, white, coeffs + axis * delta))
                .map(|v| v.to_array()),
        );

        let jacobian = (r_high - r_low) / (2.0 * delta);

        let update = jacobian.inverse() * (rgb - color);
        if update.is_nan() {
            println!("{rgb} got nan, have {color} {coeffs}");
            break;
        }

        coeffs -= update;
        let largest = coeffs.abs().max_element();
        if largest > 200.0 {
            coeffs *= 200.0 / largest;
        }
    }

    best.extend(0.0).as_vec4().to_array()
}

fn compute_color(matching: &[Vec3], white: &TableSpectrum, coeffs: DVec3) -> DVec3 {
    (360..=830)
        .zip(matching)
        .zip(&white.data)
        .map(|((wl, matching), &white)| {
            let wl = (wl as f64 - 360.0) / (830.0 - 360.0);
            let x = coeffs.x * wl * wl + coeffs.y * wl + coeffs.z;
            let v = 0.5 + x / (2.0 * (1.0 + x * x).sqrt());
            v * white as f64 * matching.as_dvec3()
        })
        .sum::<DVec3>()
}
