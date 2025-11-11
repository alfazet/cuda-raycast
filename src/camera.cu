#include "camera.cuh"

Camera::Camera(int width, int height, float fov, float near, float far) : m_viewportWidth{width},
                                                                          m_viewportHeight{height}, m_fov{fov},
                                                                          m_near{near}, m_far{far}
{
    this->pos = v3(0.0f, 0.0f, 3.0f);
    this->m_speed = 2.0f;
    this->m_rotSpeed = 0.005f;
    this->m_forwardDir = v3(0.0f, 0.0f, -1.0f);
    this->m_rightDir = glm::normalize(glm::cross(this->m_forwardDir, worldUpDir));
    this->m_upDir = glm::normalize(glm::cross(this->m_rightDir, this->m_forwardDir));
    this->setProjMatrix();
    this->setViewMatrix();
}

void Camera::handleKey(int key, float dt)
{
    switch (key)
    {
    case GLFW_KEY_W:
        this->pos += this->m_forwardDir * dt * this->m_speed;
        break;
    case GLFW_KEY_S:
        this->pos -= this->m_forwardDir * dt * this->m_speed;
        break;
    case GLFW_KEY_D:
        this->pos += this->m_rightDir * dt * this->m_speed;
        break;
    case GLFW_KEY_A:
        this->pos -= this->m_rightDir * dt * this->m_speed;
        break;
    case GLFW_KEY_Q:
        this->pos += this->m_upDir * dt * this->m_speed;
        break;
    case GLFW_KEY_E:
        this->pos -= this->m_upDir * dt * this->m_speed;
        break;
    }
    this->setViewMatrix();
}

void Camera::handleMouse(v2 delta)
{
    float dPitch = delta.y * this->m_rotSpeed;
    float dYaw = delta.x * this->m_rotSpeed;

    glm::quat q = glm::normalize(glm::cross(glm::angleAxis(dPitch, this->m_rightDir),
                                            glm::angleAxis(-dYaw, worldUpDir)));

    this->m_forwardDir = glm::normalize(glm::rotate(q, this->m_forwardDir));
    this->m_rightDir = glm::normalize(glm::cross(this->m_forwardDir, worldUpDir));
    this->m_upDir = glm::normalize(glm::cross(this->m_rightDir, this->m_forwardDir));
}

// call on viewport change
void Camera::setProjMatrix()
{
    this->projM = glm::perspective(this->m_fov, this->m_viewportWidth / static_cast<float>(this->m_viewportHeight),
                                   this->m_near, this->m_far);
    this->invProjM = glm::inverse(this->projM);
}

// call on position change
void Camera::setViewMatrix()
{
    this->viewM = glm::lookAt(this->pos, this->pos + this->m_forwardDir, worldUpDir);
    this->invViewM = glm::inverse(this->viewM);
}

void Camera::setRayDirections()
{
    this->setViewMatrix();
    this->rayDirs.resize(this->m_viewportWidth * this->m_viewportHeight);
    for (int i = 0; i < this->m_viewportHeight; i++)
    {
        for (int j = 0; j < this->m_viewportWidth; j++)
        {
            v2 p(j / static_cast<float>(this->m_viewportWidth), i / static_cast<float>(this->m_viewportHeight));
            p = 2.0f * p - 1.0f;
            v4 target = this->invProjM * v4(p.x, p.y, 1.0f, 1.0f);
            this->rayDirs[i * this->m_viewportWidth + j] =
                v3(this->invViewM * v4(glm::normalize(v3(target) / target.w), 0.0f));
        }
    }

}