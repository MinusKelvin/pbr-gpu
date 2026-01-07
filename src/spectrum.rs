use std::error::Error;
use std::path::Path;

use glam::FloatExt;
use ordered_float::OrderedFloat;
use wgpu::util::DeviceExt;

pub fn load_spectrums(device: &wgpu::Device) -> wgpu::Buffer {
    let data = load_data()
        .inspect_err(|e| eprintln!("Error loading spectra: {e}"))
        .unwrap();

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("spectra"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn load_data() -> Result<Vec<f32>, Box<dyn Error>> {
    // scale XYZ such that 1 W of 555nm light is 683.002 nits
    let [x, y, z] = load_spectrum("spectrum/CIE_xyz_1931_2deg.csv", 683.002)?;
    let y_int = y.iter().sum::<f32>();
    // scale D65 to 1 nit brightness. standard D65 is scaled such that int(D65*Y) = 100
    let [d65] = load_spectrum("spectrum/CIE_std_illum_D65.csv", 1.0 / (y_int * 100.0))?;

    let mut result = vec![];
    result.extend(x);
    result.extend(y);
    result.extend(z);
    result.extend(d65);
    Ok(result)
}

fn load_spectrum<const N: usize>(path: &str, scale: f32) -> Result<[Vec<f32>; N], Box<dyn Error>> {
    let mut piecewise = annotate_path(path.as_ref(), load_csv::<N>)?;
    piecewise.iter_mut().for_each(|(_, v)| *v = v.map(|v| v * scale));
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

fn piecewise_to_densely_sampled<const N: usize>(f: Vec<(f32, [f32; N])>) -> [Vec<f32>; N] {
    (0..N)
        .map(|j| {
            (360..=830)
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
                .collect()
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}
