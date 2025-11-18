#ifndef CUDA_RAYCAST_CAMERA_CUH
#define CUDA_RAYCAST_CAMERA_CUH

#include "common.cuh"

class Camera
{
public:
    static constexpr float3 cameraOrigin = float3(0.0f, 0.0f, 5.0f);
    float3 pos, forwardDir = float3(0.0f, 0.0f, -1.0f);

    Camera();

    void handleKey(int key, float dt);

private:
    float m_speed;

    void setDirections();
};

#endif //CUDA_RAYCAST_CAMERA_CUH