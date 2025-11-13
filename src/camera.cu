#include "camera.cuh"

Camera::Camera(v3 pos, float pitch, float yaw, float fov) : pos{pos}, pitch{pitch}, yaw{yaw}, fov{fov}
{
    this->m_speed = 2.0f;
    this->m_rotSpeed = 1.0f;
    this->setDirections();
}

void Camera::setDirections()
{
    this->forward = glm::normalize(v3(-sinf(this->yaw) * cosf(this->pitch), sinf(this->pitch),
                                      -cosf(this->yaw) * cosf(this->pitch)));
    this->right = glm::normalize(v3(cosf(this->yaw), 0.0f, -sinf(this->yaw)));
    this->up = glm::normalize(glm::cross(this->right, this->forward));
}

void Camera::handleKey(int key, float dt)
{
    switch (key)
    {
    case GLFW_KEY_W:
        this->pos += this->forward * this->m_speed * dt;
        break;
    case GLFW_KEY_S:
        this->pos -= this->forward * this->m_speed * dt;
        break;
    case GLFW_KEY_A:
        this->pos -= this->right * this->m_speed * dt;
        break;
    case GLFW_KEY_D:
        this->pos += this->right * this->m_speed * dt;
        break;
    case GLFW_KEY_Q:
        this->pos -= this->up * this->m_speed * dt;
        break;
    case GLFW_KEY_E:
        this->pos += this->up * this->m_speed * dt;
        break;
    case GLFW_KEY_UP:
        this->pitch += this->m_rotSpeed * dt;
        break;
    case GLFW_KEY_DOWN:
        this->pitch -= this->m_rotSpeed * dt;
        break;
    case GLFW_KEY_LEFT:
        this->yaw += this->m_rotSpeed * dt;
        break;
    case GLFW_KEY_RIGHT:
        this->yaw -= this->m_rotSpeed * dt;
        break;
    }
    this->setDirections();
}
