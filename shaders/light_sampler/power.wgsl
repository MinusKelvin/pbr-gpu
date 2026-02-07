#import /light.wgsl

@group(0) @binding(228)
var<storage> POWER_LS_DATA: array<PlsAliasBucket>;

struct PowerLightSampler {
    base: u32,
    count: u32,
}

struct PlsAliasBucket {
    light: LightId,
    pmf: f32,
    q: f32,
    other: u32,
}

fn light_sampler_power_sample(ls: PowerLightSampler, ref_p: vec3f, random: f32) -> LightIdSample {
    if ls.count == 0 {
        return LightIdSample();
    }

    var idx = u32(random * f32(ls.count));
    let u = fract(random * f32(ls.count));

    if u >= POWER_LS_DATA[idx + ls.base].q {
        idx = POWER_LS_DATA[idx + ls.base].other;
    }

    let light = POWER_LS_DATA[idx + ls.base].light;
    let pmf = POWER_LS_DATA[idx + ls.base].pmf;

    return LightIdSample(light, pmf);
}

fn light_sampler_power_pmf(ls: PowerLightSampler, ref_p: vec3f, light: LightId) -> f32 {
    let path = light_sample_path(light);
    if path >= ls.count {
        return 0;
    }
    if POWER_LS_DATA[path + ls.base].light.id != light.id {
        return 0;
    }
    return POWER_LS_DATA[path + ls.base].pmf;
}
