#ifndef CUDA_RAYCAST_LIGHT_CUH
#define CUDA_RAYCAST_LIGHT_CUH

#include "common.cuh"

struct Light
{
    v3 pos;
    v3 color;
};

#endif //CUDA_RAYCAST_LIGHT_CUH