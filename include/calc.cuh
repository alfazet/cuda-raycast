#ifndef CUDA_RAYCAST_CALC_CUH
#define CUDA_RAYCAST_CALC_CUH

#include "common.cuh"

constexpr float EPS = std::numeric_limits<float>::epsilon();

// triangle a-b-c, with normals na, nb and nc
// in those points and texture coords u{a, b, c}, v{a, b, c}
// whenever we register that a ray hit this triangle,
// we find the bary coordinates of that point
// and interpolate the normals based on them
struct Triangle
{
    v3 a, b, c;
    v2 uva, uvb, uvc;
    v3 na, nb, nc;
};

namespace calc
{
// return t such that P = origin + t * ray_dir is the point where the ray from origin along ray_dir
// intersects the given triangle
// if there's no intersection, return 0
// (implements the Möller-Trumbore algorithm)
__device__ float triangleIntersection(const v3& origin, const v3& ray_dir, const Triangle& triangle);
}

#endif //CUDA_RAYCAST_CALC_CUH