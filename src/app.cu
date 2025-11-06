#include "app.cuh"

void framebufferSizeCallback(GLFWwindow* window, int new_width, int new_height)
{
    App& app = *static_cast<App*>(glfwGetWindowUserPointer(window));
    app.width = new_width;
    app.height = new_height;
    // app.resize
    glViewport(0, 0, new_width, new_height);
}

void cursorPosCallback(GLFWwindow* window, double posX, double posY)
{
    App& app = *static_cast<App*>(glfwGetWindowUserPointer(window));
    if (app.m_justSpawned)
    {
        app.m_mouseX = posX;
        app.m_mouseY = posY;
        app.m_justSpawned = false;
    }
    else
    {
        double dX = posX - app.m_mouseX;
        double dY = app.m_mouseY - posY;
        app.m_mouseX = posX;
        app.m_mouseY = posY;
        app.handle_mouse(glm::vec2(dX, dY));
    }
}

App::App() : width{DEFAULT_WIN_HEIGHT}, height{DEFAULT_WIN_HEIGHT}, m_mouseX{width / 2.0}, m_mouseY{height / 2.0}
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    this->window = glfwCreateWindow(this->width, this->height, "CUDA Raycast", nullptr, nullptr);
    if (this->window == nullptr)
    {
        ERR_AND_DIE("glfwCreateWindow failed");
    }
    glfwMakeContextCurrent(this->window);
    glfwSetInputMode(this->window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    glfwSwapInterval(0);
    if (!gladLoadGL(glfwGetProcAddress))
    {
        ERR_AND_DIE("gladLoadGL failed");
    }
    glDepthFunc(GL_LESS);
    glViewport(0, 0, this->width, this->height);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_ImplGlfw_InitForOpenGL(this->window, true);
    ImGui_ImplOpenGL3_Init();

    glfwSetWindowUserPointer(this->window, this);
    glfwSetFramebufferSizeCallback(this->window, framebufferSizeCallback);
    glfwSetCursorPosCallback(this->window, cursorPosCallback);
}

App::~App()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
}

void App::run()
{
    int fps = 0, framesThisSecond = 0;
    double prevTime = glfwGetTime();
    while (!glfwWindowShouldClose(this->window))
    {
        this->handle_keys();

        double curTime = glfwGetTime();
        framesThisSecond++;
        // check how many frames we rendered over the last second
        // (so the current FPS)
        if (curTime - prevTime >= 1.0)
        {
            fps = framesThisSecond;
            framesThisSecond = 0;
            prevTime += curTime;
        }

        glClearColor(0.5, 0.0, 0.25, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        this->render_imgui_frame(fps);
        glfwSwapBuffers(this->window);
        glfwPollEvents();
    }
}

void App::render_imgui_frame(const int fps)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (!ImGui::Begin("Menu", nullptr, 0))
    {
        ImGui::End();
        return;
    }

    ImGui::PushItemWidth(0.5f * -ImGui::GetWindowWidth());
    ImGui::Text("%d FPS", fps);
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}


void App::handle_key(int key)
{
    (void)key;
}

void App::handle_keys()
{
    if (glfwGetKey(this->window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(this->window, true);
    }

    for (const auto boundKey : BOUND_KEYS)
    {
        if (glfwGetKey(this->window, boundKey) == GLFW_PRESS)
        {
            this->handle_key(boundKey);
            break;
        }
    }
}

void App::handle_mouse(glm::vec2 delta)
{
    (void)delta;
}
