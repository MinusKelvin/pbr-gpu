@group(0) @binding(192)
var<storage> FLOAT_DATA: array<f32>;

struct TableSampler1d {
    min_x: f32,
    max_x: f32,
    cdf_ptr: u32,
    len: u32,
}

struct TableSample1d {
    value: f32,
    pdf: f32,
    index: u32,
}

struct TablePdf1d {
    pdf: f32,
    index: u32,
}

struct TableSampler2d {
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
    cdf_ptr: u32,
    width: u32,
    height: u32,
}

struct TableSample2d {
    value: vec2f,
    pdf: f32,
}

fn table_1d_sample(table: TableSampler1d, random: f32) -> TableSample1d {
    let integral = FLOAT_DATA[table.cdf_ptr + table.len];
    let area = table.max_x - table.min_x;
    if integral == 0.0 {
        return TableSample1d();
    }

    let u = random * integral;
    var min = 0u;
    var max = table.len;
    while min < max {
        let mid = (min + max + 1) / 2;
        if FLOAT_DATA[table.cdf_ptr + mid] <= u {
            min = mid;
        } else {
            max = mid - 1u;
        }
    }

    let v0 = FLOAT_DATA[table.cdf_ptr + min];
    let v1 = FLOAT_DATA[table.cdf_ptr + min + 1];
    let x = f32(min) + (u - v0) / (v1 - v0);
    return TableSample1d(
        x / f32(table.len) * area + table.min_x,
        (v1 - v0) * f32(table.len) / integral / area,
        min,
    );
}

fn table_1d_pdf(table: TableSampler1d, x: f32) -> TablePdf1d {
    let integral = FLOAT_DATA[table.cdf_ptr + table.len];
    if integral == 0 {
        return TablePdf1d();
    }
    let area = table.max_x - table.min_x;

    let min = u32((x - table.min_x) / area * f32(table.len));

    let v0 = FLOAT_DATA[table.cdf_ptr + min];
    let v1 = FLOAT_DATA[table.cdf_ptr + min + 1];

    let pdf = (v1 - v0) * f32(table.len) / integral / area;
    return TablePdf1d(pdf, min);
}

fn table_2d_sample(table: TableSampler2d, random: vec2f) -> TableSample2d {
    let y_sampler = TableSampler1d(
        table.min_y,
        table.max_y,
        table.cdf_ptr + (table.width + 1) * table.height,
        table.height,
    );
    let y_sample = table_1d_sample(y_sampler, random.y);
    if y_sample.pdf == 0 {
        return TableSample2d();
    }

    let x_sampler = TableSampler1d(
        table.min_x,
        table.max_x,
        table.cdf_ptr + (table.width + 1) * y_sample.index,
        table.width,
    );
    let x_sample = table_1d_sample(x_sampler, random.x);

    return TableSample2d(
        vec2f(x_sample.value, y_sample.value),
        x_sample.pdf * y_sample.pdf
    );
}

fn table_2d_pdf(table: TableSampler2d, xy: vec2f) -> f32 {
    let y_sampler = TableSampler1d(
        table.min_y,
        table.max_y,
        table.cdf_ptr + (table.width + 1) * table.height,
        table.height,
    );
    let y_pdf = table_1d_pdf(y_sampler, xy.y);

    let x_sampler = TableSampler1d(
        table.min_x,
        table.max_x,
        table.cdf_ptr + (table.width + 1) * y_pdf.index,
        table.width,
    );
    let x_pdf = table_1d_pdf(x_sampler, xy.x);

    return x_pdf.pdf * y_pdf.pdf;
}
