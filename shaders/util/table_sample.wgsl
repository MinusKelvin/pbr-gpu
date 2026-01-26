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

fn sample_table_1d(table: TableSampler1d, random: f32) -> TableSample1d {
    let integral = FLOAT_DATA[table.cdf_ptr + table.len];
    let area = table.max_x - table.min_x;
    if integral == 0.0 {
        return TableSample1d(
            random * area + table.min_x,
            1.0 / area,
            u32(random * f32(table.len))
        );
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

fn sample_table_2d(table: TableSampler2d, random: vec2f) -> TableSample2d {
    let y_sampler = TableSampler1d(
        table.min_y,
        table.max_y,
        table.cdf_ptr + (table.width + 1) * table.height,
        table.height,
    );
    let y_sample = sample_table_1d(y_sampler, random.y);

    let x_sampler = TableSampler1d(
        table.min_x,
        table.max_x,
        table.cdf_ptr + (table.width + 1) * y_sample.index,
        table.width,
    );
    let x_sample = sample_table_1d(x_sampler, random.x);

    return TableSample2d(
        vec2f(x_sample.value, y_sample.value),
        x_sample.pdf * y_sample.pdf
    );
}
