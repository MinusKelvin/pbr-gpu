#import /light.wgsl

@group(0) @binding(226)
var<storage> UNIFORM_LS_DATA: array<LightId>;

struct UniformLightSampler {
    base: u32,
    count: u32,
}

fn light_sampler_uniform_sample(ls: UniformLightSampler, ref_p: vec3f, random: f32) -> LightIdSample {
    if ls.count == 0 {
        return LightIdSample();
    }
    let light = UNIFORM_LS_DATA[u32(random * f32(ls.count)) + ls.base];
    return LightIdSample(light, 1 / f32(ls.count));
}

fn light_sampler_uniform_pmf(ls: UniformLightSampler, ref_p: vec3f, light: LightId) -> f32 {
    let path = light_sample_path(light);
    if path >= ls.count {
        return 0;
    }
    if UNIFORM_LS_DATA[path + ls.base].id != light.id {
        return 0;
    }
    return 1 / f32(ls.count);
}
