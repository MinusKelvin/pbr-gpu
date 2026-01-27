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

# alternative

We could manually impl the stack for traversal.
That way the TLAS and BLAS both use the same stack for acceleration structure traversal.
Additionally, transform nodes can appear, which makes the whole BLAS thing more possible.
It also allows instancing-within-instancing, which is cool.
We will need to implement a similar system for mix materials and mix textures, if we ever add support for those.
The only wrinkle is that arbitrarily composing these kinds of things makes lights in instanced objects even more difficult, since every instancing level requires an additional light sampling path to be traversed to find the light sampling probability for MIS.
One issue is maintaining the stack for transform nodes, which require saving the original ray since we don't want to lose precision by un-transforming the ray.

# final design

The TLAS is a BVH of Objects.

1. Simple Object
  - Does not have a transformation matrix
  - Delegates to a Primitive
2. BVH Object
  - Has a transformation matrix
  - Knows its light sampling path
  - Shares the BVH of Primitives
  - Shares the (BVH) light sampler
3. SVO Object
  - Has a transformation matrix
  - Knows its light sampling path
  - Shares the SVO of voxels
  - Shares the SVO light sampler

A Primitive:
  - Has a shape (sphere, disk, triangle, etc.)
  - Has a material
  - Has an alpha mask
  - Knows the medium on each side of it
  - Knows its light sampling path
Since most primitives only have a shape and a material, there should be a special .

Voxels are like primitives, but they don't know their light sampling path (this is in fact the position of the voxel in the SVO, which is found during SVO traversal).

# radeon gpu profiler capture

remove submits after the work, then use `MESA_VK_TRACE_PER_SUBMIT=1 MESA_VK_TRACE=rgp` env vars.
