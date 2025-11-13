#ifndef CUDA_RAYCAST_CALC_CUH
#define CUDA_RAYCAST_CALC_CUH

#include "common.cuh"

constexpr float EPS = std::numeric_limits<float>::epsilon();

// triangle a-b-c, with normals na, nb and nc in those points
struct Triangle
{
    v3 a, b, c;
    v3 na, nb, nc;
};

namespace calc
{
// return t such that P = origin + t * ray_dir is the point where the ray from origin along ray_dir
// intersects the given triangle
// if there's no intersection, return 0
// (implements the Möller-Trumbore algorithm)
__device__ float triangleIntersection(const v3& origin, const v3& dir, const Triangle& triangle);
}

#endif //CUDA_RAYCAST_CALC_CUH