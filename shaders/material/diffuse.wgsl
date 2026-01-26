#import /material.wgsl
#import /texture.wgsl
#import /spectrum.wgsl

struct DiffuseMaterial {
    texture: TextureId
}

fn material_diffuse_evaluate(material: DiffuseMaterial, uv: vec2f, wl: Wavelengths) -> Bsdf {
    var bsdf: Bsdf;
    bsdf.id = BSDF_DIFFUSE;
    bsdf.v0 = texture_evaluate(material.texture, uv, wl);
    return bsdf;
}

fn bsdf_diffuse_f(bsdf: Bsdf, wo: vec3f, wi: vec3f) -> vec4f {
    return vec4f(wo.z * wi.z > 0) * bsdf.v0 / PI;
}

fn bsdf_diffuse_sample(bsdf: Bsdf, wi: vec3f, random: vec3f) -> BsdfSample {
    let disk = sample_uniform_disk(random.xy);
    let z = sqrt(1 - dot(disk, disk));
    return BsdfSample(
        bsdf.v0 / PI,
        vec3f(disk, copysign(z, wi.z)),
        z / PI,
    );
}
