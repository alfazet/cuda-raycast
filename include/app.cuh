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
    constexpr static int TEXTURE_WIDTH = 1920;
    constexpr static int TEXTURE_HEIGHT = 1080;
    constexpr static int BOUND_KEYS[4] = {GLFW_KEY_W, GLFW_KEY_S, GLFW_KEY_A, GLFW_KEY_D};

    int width, height;
    GLFWwindow* window;
    Shader* shader;

    App();

    ~App();

    void run();

    void resize(int width, int height);

    void handle_key(int key);

    void handle_keys();

    void handle_mouse(glm::vec2 delta);

    void render_imgui_frame(int fps);

    friend void cursorPosCallback(GLFWwindow* window, double posX, double posY);

private:
    double m_mouseX, m_mouseY;
    bool m_justSpawned = true;
};

#endif //CUDA_RAYCAST_APP_CUH
