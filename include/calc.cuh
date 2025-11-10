#ifndef CUDA_RAYCAST_CALC_CUH
#define CUDA_RAYCAST_CALC_CUH

#include "common.cuh"

// triangle a-b-c, with normals na, nb and nc
// in those points and texture coords u{a, b, c}, v{a, b, c}
// whenever we register that a ray hit this triangle,
// we find the bary coordinates of that point
// and interpolate the normals based on them
struct Triangle
{
    float3 a, b, c;
    float2 uva, uvb, uvc;
    glm::vec3 na, nb, nc;
};

// TODO: when passing this stuff to the GPU do it like this:
// vertices: [a1, b1, c1, a2, b2, c2, ...]
// normals: [a1n, b1n, c1n, a2n, b2n, c2n, ...]
// texture coords: [a1u, a1v, b1u, b1v, c1u, c1v, ...]

// general outline of the kernel:
// 0. pass vertices, normals, texture coords, light sources, ...
// (maybe place them in shared memory?)
// 1. create a ray from the camera to the (x, y) viewport coord of this thread
// (use orthographic projection)
// 2. for each, check if it was hit - and take the hitpoint with the smallest z-coord
// 3. do lighting computation (with interpolated normals)

#endif //CUDA_RAYCAST_CALC_CUH