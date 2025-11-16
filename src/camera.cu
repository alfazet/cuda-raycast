#include "camera.cuh"

Camera::Camera(float fov) : pos{cameraOrigin}, pitch{0.0f}, yaw{0.0f}, fov{fov}
{
    this->m_speed = 60.0f;
    this->m_rotSpeed = 1.0f;
    this->setDirections();
}

void Camera::setDirections()
{
    quat orientation = quat(v3(this->pitch, this->yaw, 0.0f));
    this->forward = glm::normalize(orientation * worldForward);
    this->right = glm::normalize(glm::cross(this->forward, worldUp));
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
    case GLFW_KEY_0:
        this->pitch = 0.0f;
        this->yaw = 0.0f;
        this->pos = cameraOrigin;
        break;
    }
    this->setDirections();
}
