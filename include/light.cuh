#ifndef CUDA_RAYCAST_LIGHT_CUH
#define CUDA_RAYCAST_LIGHT_CUH

#include "common.cuh"

struct Light
{
    float3 pos, color;
};

struct LightSOA
{
    float3 *pos, *color;
};

#endif //CUDA_RAYCAST_LIGHT_CUH