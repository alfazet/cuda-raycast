#ifndef CUDA_RAYCAST_RENDERER_CUH
#define CUDA_RAYCAST_RENDERER_CUH

#include "calc.cuh"
#include "common.cuh"

class Renderer
{
public:
    Renderer(uint pbo, int width, int height, std::vector<Triangle>& faces);

    ~Renderer();

    void render();

private:
    cudaGraphicsResource* m_pboRes{};
    int m_width, m_height, m_nFaces;
    void* m_dTexBuf; // d -> stored on the device
    Triangle* m_dFaces;
    dim3 m_blockDim, m_gridDim;
};

#endif //CUDA_RAYCAST_RENDERER_CUH