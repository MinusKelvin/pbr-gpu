#import /material.wgsl
#import /texture.wgsl
#import /spectrum.wgsl

struct DiffuseTransmitMaterial {
    normal_map: u32,
    bump_map: FloatTextureId,
    reflectance: SpectrumTextureId,
    transmittance: SpectrumTextureId,
    scale: FloatTextureId,
}

fn material_diffuse_transmit_evaluate(material: DiffuseTransmitMaterial, uv: vec2f, wl: Wavelengths) -> BsdfParams {
    var bsdf: BsdfParams;
    bsdf.id = BSDF_DIFFUSE_TRANSMIT;
    let scale = float_texture_evaluate(material.scale, uv);
    bsdf.v0 = spectrum_texture_evaluate(material.reflectance, uv, wl) * scale;
    bsdf.v1 = spectrum_texture_evaluate(material.transmittance, uv, wl) * scale;
    return bsdf;
}

fn bsdf_diffuse_transmit_f(bsdf: BsdfParams, wo: vec3f, wi: vec3f) -> vec4f {
    return select(
        bsdf.v1,
        bsdf.v0,
        wo.z * wi.z > 0
    ) / PI;
}
