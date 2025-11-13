#ifndef CUDA_RAYCAST_CAMERA_CUH
#define CUDA_RAYCAST_CAMERA_CUH

#include "common.cuh"

class Camera
{
public:
    static constexpr v3 worldForward = v3(0.0f, 0.0f, -1.0f), worldRight = v3(1.0f, 0.0f, 0.0f), worldUp = v3(
            0.0f, 1.0f, 0.0f);
    float pitch, yaw, fov;
    v3 pos, forward, right, up;

    Camera() = default;

    Camera(v3 pos, float pitch, float yaw, float fov);

    void handleKey(int key, float dt);

private:
    float m_speed, m_rotSpeed;

    void setDirections();
};

#endif //CUDA_RAYCAST_CAMERA_CUH