#ifndef CUDA_RAYCAST_CAMERA_CUH
#define CUDA_RAYCAST_CAMERA_CUH

#include "common.cuh"

class Camera
{
public:
    static constexpr v3 worldUpDir = v3(0.0f, 1.0f, 0.0f);
    v3 pos, forwardDir, rightDir, upDir;

    Camera(int width, int height, float fov, float near, float far);

    void handleKey(int key, float dt);

    void handleMouse(v2 delta);

private:
    float m_speed, m_rotSpeed;
};

#endif //CUDA_RAYCAST_CAMERA_CUH