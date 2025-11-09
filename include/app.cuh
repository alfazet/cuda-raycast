#ifndef CUDA_RAYCAST_APP_CUH
#define CUDA_RAYCAST_APP_CUH

#include "common.cuh"
#include "shader.cuh"
#include "renderer.cuh"

class App
{
public:
    constexpr static int DEFAULT_WIN_WIDTH = 1024;
    constexpr static int DEFAULT_WIN_HEIGHT = 576;
    constexpr static int DEFAULT_TEXTURE_WIDTH = 2560;
    constexpr static int DEFAULT_TEXTURE_HEIGHT = 1440;
    constexpr static int BOUND_KEYS[4] = {GLFW_KEY_W, GLFW_KEY_S, GLFW_KEY_A, GLFW_KEY_D};

    int width, height;
    int texWidth, texHeight;
    GLFWwindow* window;
    Shader* shader;
    Renderer* renderer;

    App();

    ~App();

    void run();

    void handleKey(int key);

    void handleKeys();

    void handleMouse(glm::vec2 delta);

    void renderImguiFrame(int fps);

    friend void cursorPosCallback(GLFWwindow* window, double posX, double posY);

private:
    double m_mouseX, m_mouseY;
    bool m_justSpawned = true;
    uint m_vao, m_vbo, m_vboTex, m_pbo, m_tex, m_ebo;
};

#endif //CUDA_RAYCAST_APP_CUH
