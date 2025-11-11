#include "calc.cuh"

namespace calc
{
__device__ float triangleIntersection(const v3& origin, const v3& ray_dir, const Triangle& triangle)
{
    v3 e1 = triangle.b - triangle.a;
    v3 e2 = triangle.c - triangle.a;
    v3 ray_e2_cross = glm::cross(ray_dir, e2);
    float det = glm::dot(e1, ray_e2_cross);
    if (abs(det) < EPS)
    {
        return 0;
    }
    float inv_det = 1.0f / det;
    v3 s = origin - triangle.a;
    float u = inv_det * glm::dot(s, ray_e2_cross);
    if ((u < 0 && abs(u) > EPS) || (u > 1 && abs(u - 1) > EPS))
    {
        return 0;
    }
    v3 s_e1_cross = glm::cross(s, e1);
    float v = inv_det * glm::dot(ray_dir, s_e1_cross);
    if ((v < 0 && abs(v) > EPS) || (u + v > 1 && abs(u + v - 1) > EPS))
    {
        return 0;
    }
    float t = inv_det * glm::dot(e2, s_e1_cross);

    return t > EPS ? t : 0.0f;
}
}
