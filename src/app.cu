#include "app.cuh"
#include "obj_parser.cuh"

void framebufferSizeCallback(GLFWwindow* window, int new_width, int new_height)
{
    App& app = *static_cast<App*>(glfwGetWindowUserPointer(window));
    app.width = new_width;
    app.height = new_height;
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
        app.handleMouse(glm::vec2(dX, dY));
    }
}

void initBuffers(uint& vao, uint& vbo, uint& vboTex, uint& ebo)
{
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &vboTex);
    glGenBuffers(1, &ebo);

    float2 vertices[4];
    vertices[0] = float2(1.0f, 1.0f);
    vertices[1] = float2(1.0f, -1.0f);
    vertices[2] = float2(-1.0f, -1.0f);
    vertices[3] = float2(-1.0f, 1.0f);
    float2 tex_coords[4];
    tex_coords[0] = float2(1.0f, 1.0f);
    tex_coords[1] = float2(1.0f, 0.0f);
    tex_coords[2] = float2(0.0f, 0.0f);
    tex_coords[3] = float2(0.0f, 1.0f);
    // we'll render triangles 0-1-3 and 1-2-3
    // (the indices refer to the vertices/texture coordinates above)
    int indices[] = {0, 1, 3, 1, 2, 3};

    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // vertex attrib 0 (positions)
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float2), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float2), nullptr);
    glEnableVertexAttribArray(0);

    // vertex attrib 1 (tex coords)
    glBindBuffer(GL_ARRAY_BUFFER, vboTex);
    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float2), tex_coords, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float2), nullptr);
    glEnableVertexAttribArray(1);
}

void initTexture(uint& tex, uint& pbo, int width, int height)
{
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 nullptr);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(uchar3), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

App::App() : width{DEFAULT_WIN_WIDTH}, height{DEFAULT_WIN_HEIGHT}, texWidth{DEFAULT_TEXTURE_WIDTH},
             texHeight{DEFAULT_TEXTURE_HEIGHT}, m_mouseX{width / 2.0}, m_mouseY{height / 2.0}
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
    glViewport(0, 0, this->width, this->height);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
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

    std::string vert_shader_path = std::string(PROJECT_DIR) + "/shaders/shader.vert";
    std::string frag_shader_path = std::string(PROJECT_DIR) + "/shaders/shader.frag";
    this->shader = new Shader(vert_shader_path.c_str(), frag_shader_path.c_str());
    glUseProgram(this->shader->programId);

    initBuffers(this->m_vao, this->m_vbo, this->m_vboTex, this->m_ebo);
    initTexture(this->m_tex, this->m_pbo, this->texWidth, this->texHeight);
    this->renderer = new Renderer(this->m_pbo, this->texWidth, this->texHeight);
    glViewport(0, 0, this->texWidth, this->texHeight);

    ObjParser parser;
    // TODO: get file name from cli args
    parser.parseFile("../test.obj");
    this->m_faces = std::move(parser.faces);

    cudaErrCheck(cudaSetDevice(0));
}

App::~App()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glUseProgram(0);
    glDeleteProgram(this->shader->programId);
    glDeleteVertexArrays(1, &this->m_vao);
    glDeleteBuffers(1, &this->m_vbo);
    glDeleteBuffers(1, &this->m_vboTex);
    glDeleteBuffers(1, &this->m_ebo);
    glDeleteBuffers(1, &this->m_pbo);
    glDeleteTextures(1, &this->m_tex);
    glfwTerminate();
}

void App::run()
{
    int fps = 0, framesThisSecond = 0;
    double prevTime = glfwGetTime();
    while (!glfwWindowShouldClose(this->window))
    {
        this->handleKeys();

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

        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        this->renderer->render(this->m_faces);
        glBindTexture(GL_TEXTURE_2D, this->m_tex);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, this->m_pbo);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->texWidth, this->texHeight, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glBindVertexArray(this->m_vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        this->renderImguiFrame(fps);
        glfwSwapBuffers(this->window);
        glfwPollEvents();
    }
}

void App::renderImguiFrame(const int fps)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // if (!ImGui::Begin("Menu", nullptr, 0))
    // {
    //     ImGui::End();
    //     return;
    // }
    //
    // ImGui::PushItemWidth(0.33f * ImGui::GetWindowWidth());
    // ImGui::Text("%d FPS", fps);
    // ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}


void App::handleKey(int key)
{
    (void)key;
}

void App::handleKeys()
{
    if (glfwGetKey(this->window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(this->window, true);
    }

    for (const auto boundKey : BOUND_KEYS)
    {
        if (glfwGetKey(this->window, boundKey) == GLFW_PRESS)
        {
            this->handleKey(boundKey);
            break;
        }
    }
}

void App::handleMouse(glm::vec2 delta)
{
    (void)delta;
}