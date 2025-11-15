#ifndef CUDA_RAYCAST_APP_CUH
#define CUDA_RAYCAST_APP_CUH

#include "calc.cuh"
#include "common.cuh"
#include "shader.cuh"
#include "renderer.cuh"

class App
{
public:
    constexpr static int DEFAULT_WIN_WIDTH = 1024;
    constexpr static int DEFAULT_WIN_HEIGHT = 576;
    constexpr static int DEFAULT_TEXTURE_WIDTH = 1024;
    constexpr static int DEFAULT_TEXTURE_HEIGHT = 576;
    constexpr static int BOUND_KEYS[11] = {GLFW_KEY_W, GLFW_KEY_S, GLFW_KEY_A, GLFW_KEY_D, GLFW_KEY_E, GLFW_KEY_Q,
                                           GLFW_KEY_UP, GLFW_KEY_DOWN, GLFW_KEY_LEFT, GLFW_KEY_RIGHT, GLFW_KEY_0};

    int width, height;
    int texWidth, texHeight;
    GLFWwindow* window;
    Shader* shader;
    Renderer* renderer;

    App();

    ~App();

    void run();

    void handleKey(int key, float dt);

    void handleKeys();

    void renderImguiFrame(int fps);

    friend void cursorPosCallback(GLFWwindow* window, double posX, double posY);

private:
    float m_dt;
    uint m_vao, m_vbo, m_vboTex, m_pbo, m_tex, m_ebo;
};

#endif //CUDA_RAYCAST_APP_CUH
