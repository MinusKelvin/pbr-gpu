#import /sampler/meta.wgsl

struct FilterSample {
    p: vec2f,
    f: f32,
    pdf: f32,
}

fn filter_sample() -> FilterSample {
    let u = sample_pixel();
    // box filter
    return FilterSample(u - 0.5, 1, 1);
}
