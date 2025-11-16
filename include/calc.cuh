#ifndef CUDA_RAYCAST_CALC_CUH
#define CUDA_RAYCAST_CALC_CUH

#include "common.cuh"

constexpr float EPS = std::numeric_limits<float>::epsilon();

struct Normals
{
    float3 na, nb, nc;
};

// triangle a-b-c, with normals na, nb and nc in those points
struct Triangle
{
    float3 a, b, c;
};

// return t such that P = origin + t * ray_dir is the point where the ray from origin along ray_dir
// intersects the given triangle
// if there's no intersection, return 0
// (implements the Möller-Trumbore algorithm)
inline __device__ __host__ void triangleIntersection(float3 origin, float3 dir, Triangle* triangle, float* t,
                                                     float3* bary)
{
    *t = 0.0f;
    float3 e1 = triangle->b - triangle->a;
    float3 e2 = triangle->c - triangle->a;
    float3 dir_e2_cross = cross(dir, e2);
    float det = dot(e1, dir_e2_cross);
    if (abs(det) < EPS)
    {
        return;
    }
    float inv_det = 1.0f / det;
    float3 s = origin - triangle->a;
    float u = inv_det * dot(s, dir_e2_cross);
    if (u < 0.0f || u > 1.0f)
    {
        return;
    }
    float3 s_e1_cross = cross(s, e1);
    float v = inv_det * dot(dir, s_e1_cross);
    if (v < 0 || u + v > 1.0)
    {
        return;
    }
    // u and v are the barycentric coordinates of the hit point inside the triangle
    *t = max(0.0f, inv_det * dot(e2, s_e1_cross));
    *bary = make_float3(u, v, 1.0f - u - v);
}

inline __device__ __host__ uchar3 rgbFloatsToBytes(float3 color)
{
    unsigned char r = static_cast<unsigned char>(255.0 * color.x);
    unsigned char g = static_cast<unsigned char>(255.0 * color.y);
    unsigned char b = static_cast<unsigned char>(255.0 * color.z);

    return make_uchar3(r, g, b);
}

#endif //CUDA_RAYCAST_CALC_CUH