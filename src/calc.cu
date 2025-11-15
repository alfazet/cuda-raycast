#include "calc.cuh"

namespace calc
{
__device__ __host__ float triangleIntersection(const v3& origin, const v3& dir, const Triangle& triangle)
{
    v3 e1 = triangle.b - triangle.a;
    v3 e2 = triangle.c - triangle.a;
    v3 dir_e2_cross = glm::cross(dir, e2);
    float det = glm::dot(e1, dir_e2_cross);
    if (abs(det) < EPS)
    {
        return 0;
    }
    float inv_det = 1.0f / det;
    v3 s = origin - triangle.a;
    float u = inv_det * glm::dot(s, dir_e2_cross);
    if ((u < 0 && abs(u) > EPS) || (u > 1 && abs(u - 1) > EPS))
    {
        return 0;
    }
    v3 s_e1_cross = glm::cross(s, e1);
    float v = inv_det * glm::dot(dir, s_e1_cross);
    if ((v < 0 && abs(v) > EPS) || (u + v > 1 && abs(u + v - 1) > EPS))
    {
        return 0;
    }
    float t = inv_det * glm::dot(e2, s_e1_cross);

    return t > EPS ? t : 0.0f;
}

__device__ __host__ float3 barycentric(const v3& p, const Triangle& tri)
{
    v3 a = tri.a, b = tri.b, c = tri.c;
    float detInv = 1.0f / ((a.x - c.x) * (b.y - c.y) - (b.x - c.x) * (a.y - c.y));
    v2 res = detInv * m2(b.y - c.y, c.x - b.x, c.y - a.y, a.x - c.x) * v2(p.x - c.x, p.y - c.y);

    return float3(res.x, res.y, 1 - res.x - res.y);
}

__device__ __host__ v3 normalAt(const v3& p, const Triangle& tri)
{
    float3 bary = barycentric(p, tri);
    v3 normal = glm::normalize(bary.x * tri.na + bary.y * tri.nb + bary.z * tri.nc);

    return normal;
}

__device__ __host__ uchar3 colorBytes(float3 color)
{
    unsigned char r = static_cast<unsigned char>(255.0 * color.x);
    unsigned char g = static_cast<unsigned char>(255.0 * color.y);
    unsigned char b = static_cast<unsigned char>(255.0 * color.z);

    return uchar3(r, g, b);
}
}
