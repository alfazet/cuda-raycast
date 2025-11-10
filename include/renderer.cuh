#ifndef CUDA_RAYCAST_RENDERER_CUH
#define CUDA_RAYCAST_RENDERER_CUH

#include "calc.cuh"
#include "common.cuh"

class Renderer
{
public:
    Renderer(uint pbo, int width, int height);

    ~Renderer();

    void render(std::vector<Triangle>& faces);

private:
    cudaGraphicsResource* m_pboRes;
    int m_width, m_height;
    void* m_texBuf;
    dim3 m_blockDim, m_gridDim;
};

#endif //CUDA_RAYCAST_RENDERER_CUH