#ifndef CUDA_RAYCAST_CAMERA_CUH
#define CUDA_RAYCAST_CAMERA_CUH

#include "common.cuh"

class Camera
{
public:
    static constexpr v3 worldUpDir = v3(0.0f, 1.0f, 0.0f);

    v3 pos;
    std::vector<v3> rayOrigins, rayDirs;
    m4 projM, viewM, invProjM, invViewM;

    Camera(int width, int height, float fov, float near, float far);

    void setProjMatrix();

    void setViewMatrix();

    void setRayDirections();

    void handleKey(int key, float dt);

private:
    int m_viewportWidth, m_viewportHeight;
    v3 m_forwardDir, m_rightDir, m_upDir;
    float m_speed, m_rotSpeed, m_fov, m_near, m_far;
};

#endif //CUDA_RAYCAST_CAMERA_CUH