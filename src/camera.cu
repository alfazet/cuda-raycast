#include "camera.cuh"

Camera::Camera(int width, int height, float fov, float near, float far)
{
    this->pos = v3(0.0f, 0.0f, 3.0f);
    this->m_speed = 2.0f;
    this->m_rotSpeed = 0.5f;
    this->forwardDir = v3(0.0f, 0.0f, -1.0f);
    this->upDir = v3(0.0f, 1.0f, 0.0f);
    this->rightDir = v3(1.0f, 0.0f, 0.0f);
}

void Camera::handleKey(int key, float dt)
{
    float dPitch = 0.0f, dYaw = 0.0f;
    switch (key)
    {
    // camera movement
    case GLFW_KEY_W:
        this->pos += this->upDir * dt * this->m_speed;
        break;
    case GLFW_KEY_S:
        this->pos -= this->upDir * dt * this->m_speed;
        break;
    case GLFW_KEY_D:
        this->pos += this->rightDir * dt * this->m_speed;
        break;
    case GLFW_KEY_A:
        this->pos -= this->rightDir * dt * this->m_speed;
        break;
    case GLFW_KEY_Q:
        this->pos += this->forwardDir * dt * this->m_speed;
        break;
    case GLFW_KEY_E:
        this->pos -= this->forwardDir * dt * this->m_speed;
        break;

    // camera rotation
    case GLFW_KEY_H:
        dYaw = dt * this->m_rotSpeed;
        break;
    case GLFW_KEY_L:
        dYaw = -dt * this->m_rotSpeed;
        break;
    case GLFW_KEY_K:
        dPitch = dt * this->m_rotSpeed;
        break;
    case GLFW_KEY_J:
        dPitch = -dt * this->m_rotSpeed;
        break;

    }
    glm::quat q = glm::normalize(glm::cross(glm::angleAxis(dPitch, this->rightDir),
                                            glm::angleAxis(-dYaw, worldUpDir)));
    this->forwardDir = glm::normalize(glm::rotate(q, this->forwardDir));
    // this->rightDir = glm::normalize(glm::cross(this->forwardDir, worldUpDir));
    // this->upDir = glm::normalize(glm::cross(this->rightDir, this->forwardDir));
}

void Camera::handleMouse(v2 delta)
{
    (void)delta;
}