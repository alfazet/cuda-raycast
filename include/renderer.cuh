#ifndef CUDA_RAYCAST_RENDERER_CUH
#define CUDA_RAYCAST_RENDERER_CUH

#include "calc.cuh"
#include "camera.cuh"
#include "common.cuh"
#include "light.cuh"

class Renderer
{
public:
    Camera camera;

    Renderer(uint pbo, int width, int height, std::vector<Triangle>& faces, std::vector<Light>& lights);

    ~Renderer();

    void render();

    void handleKey(int key, float dt);

private:
    cudaGraphicsResource* m_pboRes{};
    int m_width, m_height, m_nFaces, m_nLights;
    void* m_dTexBuf; // d -> stored on the device
    Triangle* m_dFaces;
    Light* m_dLights;
    dim3 m_blockDim, m_gridDim;
};

#endif //CUDA_RAYCAST_RENDERER_CUH