#include "camera.cuh"

Camera::Camera(int width, int height, float fov, float near, float far)
{
    this->pos = v3(0.0f, 0.0f, 3.0f);
    this->shift = v3(0.0f, 0.0f, 0.0f);
    this->scale = 1.0f;
    this->m_speed = 2.0f;
    this->m_rotSpeed = 0.5f;
    this->forwardDir = v3(0.0f, 0.0f, -1.0f);
    this->upDir = v3(0.0f, 1.0f, 0.0f);
    this->rightDir = v3(1.0f, 0.0f, 0.0f);
}

void Camera::handleKey(int key, float dt)
{
    switch (key)
    {
    case GLFW_KEY_W:
        this->shift -= this->upDir * dt * this->m_speed;
        break;
    case GLFW_KEY_S:
        this->shift += this->upDir * dt * this->m_speed;
        break;
    case GLFW_KEY_D:
        this->shift -= this->rightDir * dt * this->m_speed;
        break;
    case GLFW_KEY_A:
        this->shift += this->rightDir * dt * this->m_speed;
        break;
    case GLFW_KEY_E:
        this->scale -= dt * this->m_speed;
        if (this->scale < 0.25f)
        {
            this->scale = 0.25f;
        }
        break;
    case GLFW_KEY_Q:
        this->scale += dt * this->m_speed;
        break;
    }
}

void Camera::handleMouse(v2 delta)
{
    (void)delta;
}