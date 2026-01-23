#import /material.wgsl
#import /texture.wgsl
#import /spectrum.wgsl

struct DiffuseMaterial {
    texture: TextureId
}

fn material_diffuse_evaluate(material: DiffuseMaterial, wl: Wavelengths) -> Bsdf {
    var bsdf: Bsdf;
    bsdf.id = BSDF_DIFFUSE;
    bsdf.v0 = texture_evaluate(material.texture, wl);
    return bsdf;
}

fn bsdf_diffuse_f(bsdf: Bsdf, wo: vec3f, wi: vec3f) -> vec4f {
    return vec4f(wo.z * wi.z > 0) * bsdf.v0 / PI;
}
