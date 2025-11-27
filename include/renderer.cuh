#ifndef CUDA_RAYCAST_RENDERER_CUH
#define CUDA_RAYCAST_RENDERER_CUH

#include "calc.cuh"
#include "camera.cuh"
#include "common.cuh"

__device__ constexpr float3 ZERO = float3(0.0f, 0.0f, 0.0f);
__device__ constexpr float3 ONE = float3(1.0f, 1.0f, 1.0f);
__device__ constexpr uchar3 BKG_COLOR = uchar3(32, 32, 32);
// profiling with nvprof showed no difference between 1024,
// 512 and 256 threads/block (on a 1050ti GPU with 768 CUDA cores)
constexpr dim3 CUDA_BLOCK_DIM_1D = dim3(1024, 1, 1);
constexpr dim3 CUDA_BLOCK_DIM_2D = dim3(32, 32, 1);

class Renderer
{
public:
    Camera camera = Camera();

    Renderer(int width, int height, int nFaces, int nLights, float3 color, float kD, float kS, float kA, float alpha);

    virtual ~Renderer() = default;

    virtual void render() = 0;

    void handleKey(int key, float dt);

protected:
    int m_width, m_height, m_nFaces, m_nLights;
    float m_kD, m_kS, m_kA, m_alpha, m_rotSpeed = 1.0f, m_lightAngle = 0.0f, m_scale = 1.0f, m_scaleSpeed = 0.2f;
    float3 m_color, m_angles = float3(0.0f, 0.0f, 0.0f); // angles = (yaw, pitch, roll)
};

class RendererGPU : public Renderer
{
public:
    RendererGPU(uint pbo, int width, int height, std::vector<Triangle>& faces, std::vector<Normals>& normals,
                std::vector<Light>& lights,
                float3 color, float kD, float kS, float kA, float alpha);

    ~RendererGPU() override;

    void render() override;

private:
    cudaGraphicsResource* m_pboRes{};
    void* m_dTexBuf; // d -> stored on the device
    TriangleSOA m_dFaces, m_dOriginalFaces;
    LightSOA m_dLights, m_dOriginalLights;
    NormalsSOA m_dNormals, m_dOriginalNormals;
};

class RendererCPU : public Renderer
{
public:
    RendererCPU(int width, int height, std::vector<Triangle>& faces, std::vector<Normals>& normals,
                std::vector<Light>& lights,
                float3 color, float kD, float kS, float kA, float alpha);

    ~RendererCPU() override;

    void render() override;

private:
    std::vector<uchar3> m_texBuf;
    TriangleSOA m_faces, m_originalFaces;
    LightSOA m_lights, m_originalLights;
    NormalsSOA m_normals, m_originalNormals;
};

#endif //CUDA_RAYCAST_RENDERER_CUH