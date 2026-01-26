use bytemuck::NoUninit;

use crate::scene::Scene;

impl Scene {
    pub fn add_1d_table_sampler(&mut self, min_x: f32, max_x: f32, f: &[f32]) -> TableSampler1d {
        let mut cdf = vec![0.0; f.len() + 1];
        for i in 0..f.len() {
            cdf[i + 1] = cdf[i] + f[i].abs();
        }
        let cdf_ptr = self.add_float_data(&cdf);
        TableSampler1d {
            min_x,
            max_x,
            cdf_ptr,
            len: f.len() as u32,
        }
    }

    pub fn add_2d_table_sampler(
        &mut self,
        min_x: f32,
        max_x: f32,
        min_y: f32,
        max_y: f32,
        width: u32,
        height: u32,
        f: &[f32],
    ) -> TableSampler2d {
        let width = width as usize;
        let height = height as usize;
        assert_eq!(width * height, f.len());
        let oned_size = (width + 1) * height;

        let mut cdfs = vec![0.0; (oned_size + height + 1) as usize];
        for (cdf, f) in cdfs.chunks_mut(width + 1).zip(f.chunks(width)) {
            for i in 0..f.len() {
                cdf[i + 1] = cdf[i] + f[i].abs();
            }
        }
        for y in 0..height {
            cdfs[oned_size + y + 1] = cdfs[oned_size + y] + cdfs[y * (width + 1) + width];
        }

        let cdf_ptr = self.add_float_data(&cdfs);
        TableSampler2d {
            min_x,
            max_x,
            min_y,
            max_y,
            cdf_ptr,
            width: width as u32,
            height: height as u32,
        }
    }
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct TableSampler1d {
    min_x: f32,
    max_x: f32,
    cdf_ptr: u32,
    len: u32,
}

#[derive(Copy, Clone, Debug, NoUninit)]
#[repr(C)]
pub struct TableSampler2d {
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
    cdf_ptr: u32,
    width: u32,
    height: u32,
}
