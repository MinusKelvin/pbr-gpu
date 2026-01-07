# scene structure

the idea is this:
`Shape`s:
- sphere
- disk
- triangle
- voxel
- etc.
`Primitive`s consist of a `Shape` and material/light information
`Object`s consist of one or more `Primitive`s stored in an acceleration structure (BVH/SVO)
`Instance`s consist of an `Object` and a `Transform`
the scene contains numerous `Instance`s stored in an acceleration structure

as such, the scene requires:
1. a list of BVH nodes for the instances
2. a list of instances (bvhnode, transform pair)
3. a list of BVH nodes for the primitives
4. a list of primitives (shape, material, light, etc.)
5. the shape context, which maps shape id -> concrete shape
6. the material context, which maps material id -> concrete material
7. the medium context, which maps medium id -> concrete medium

One of the capabilities we need is the ability to compute the PDF of light sampling an arbitrary point on a shape.
Interactions should therefore track not only which primitive they hit, but which instance that primitive belonged to.
1. Ask the scene light sampler for the PMF of sampling the instance
    - requires knowing the path to the instance in the BVH, which we obtain during traversal
2. Ask the instance light sampler for the PMF of sampling the primitive
    - requires knowing the path to the instance in the BVH/SVO, which we obtain during traversal
3. Ask the shape for the PDF of sampling the point

# radeon gpu profiler capture

remove submits after the work, then use `MESA_VK_TRACE_PER_SUBMIT=1 MESA_VK_TRACE=rgp` env vars.
