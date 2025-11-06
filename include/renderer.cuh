#ifndef CUDA_RAYCAST_RENDERER_CUH
#define CUDA_RAYCAST_RENDERER_CUH

#include "common.cuh"

class Renderer
{
public:
    Renderer(int width, int height);

    ~Renderer();
};

#endif //CUDA_RAYCAST_RENDERER_CUH