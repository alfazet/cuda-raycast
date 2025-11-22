#include "camera.cuh"

Camera::Camera() : pos{cameraOrigin}
{
    this->m_speed = 2.0f;
}

void Camera::handleKey(int key, float dt)
{
    switch (key)
    {
    case GLFW_KEY_UP:
        this->pos.y += this->m_speed * dt;
        break;
    case GLFW_KEY_DOWN:
        this->pos.y -= this->m_speed * dt;
        break;
    case GLFW_KEY_LEFT:
        this->pos.x -= this->m_speed * dt;
        break;
    case GLFW_KEY_RIGHT:
        this->pos.x += this->m_speed * dt;
        break;
    case GLFW_KEY_0:
        this->pos = cameraOrigin;
        break;
    }
}
