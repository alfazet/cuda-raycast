#ifndef CUDA_RAYCAST_RENDERER_CUH
#define CUDA_RAYCAST_RENDERER_CUH

#include "calc.cuh"
#include "camera.cuh"
#include "common.cuh"
#include "light.cuh"

__device__ constexpr float3 ZERO = float3(0.0f, 0.0f, 0.0f);
__device__ constexpr float3 ONE = float3(1.0f, 1.0f, 1.0f);
__device__ constexpr uchar3 BKG_COLOR = uchar3(32, 32, 32);
constexpr dim3 CUDA_BLOCK_DIM_1D = dim3(1024, 1, 1);
constexpr dim3 CUDA_BLOCK_DIM_2D = dim3(32, 32, 1);

class Renderer
{
public:
    Camera camera = Camera();

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
    float m_kD, m_kS, m_kA, m_alpha, m_rotSpeed = 1.0f, m_lightAngle = 0.0f, m_scale = 1.0f, m_scaleSpeed = 0.25f;
    float3 m_color, m_angles = float3(0.0f, 0.0f, 0.0f); // angles = (yaw, pitch, roll)
    Triangle *m_dFaces, *m_dOriginalFaces;
    Light *m_dLights, *m_dOriginalLights;
    Normals *m_dNormals, *m_dOriginalNormals;
};

#endif //CUDA_RAYCAST_RENDERER_CUH