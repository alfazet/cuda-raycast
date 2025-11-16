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

    Renderer(uint pbo, int width, int height, std::vector<Triangle>& faces, std::vector<Normals>& normals,
             std::vector<Light>& lights,
             float3 color, float kD, float kS, float kA, float alpha);

    ~Renderer();

    void render();

    void handleKey(int key, float dt);

private:
    cudaGraphicsResource* m_pboRes{};
    int m_width, m_height, m_nFaces, m_nLights;
    void* m_dTexBuf; // d -> stored on the device
    float m_kD, m_kS, m_kA, m_alpha;
    float3 m_color;
    Triangle* m_dFaces;
    Light* m_dLights;
    Normals* m_dNormals;
    dim3 m_blockDim, m_gridDim;
};

#endif //CUDA_RAYCAST_RENDERER_CUH