#import /material.wgsl
#import /texture.wgsl
#import /spectrum.wgsl

struct DiffuseTransmitMaterial {
    normal_map: u32,
    reflectance: TextureId,
    transmittance: TextureId,
    scale: TextureId,
}

fn material_diffuse_transmit_evaluate(material: DiffuseTransmitMaterial, uv: vec2f, wl: Wavelengths) -> BsdfParams {
    var bsdf: BsdfParams;
    bsdf.id = BSDF_DIFFUSE_TRANSMIT;
    let scale = texture_evaluate(material.scale, uv, wl);
    bsdf.v0 = texture_evaluate(material.reflectance, uv, wl) * scale;
    bsdf.v1 = texture_evaluate(material.transmittance, uv, wl) * scale;
    return bsdf;
}

fn bsdf_diffuse_transmit_f(bsdf: BsdfParams, wo: vec3f, wi: vec3f) -> vec4f {
    return select(
        bsdf.v1,
        bsdf.v0,
        wo.z * wi.z > 0
    ) / PI;
}
