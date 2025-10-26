#ifndef ITS_RAINING_SPHERES_COMMON_CUH
#define ITS_RAINING_SPHERES_COMMON_CUH

#include <iostream>
#include <format>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "glad/gl.h"
#define GLFW_INCLUDE_GLU 1
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#define GLM_FORCE_CUDA

#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#define ERR_AND_DIE(reason)                                                         \
{                                                                                   \
    std::cerr << std::format("fatal error in {}, line {}\n", __FILE__, __LINE__);   \
    std::cerr << std::format("reason: {}\n", (reason));                             \
    glfwTerminate();                                                                \
    exit(EXIT_FAILURE);                                                             \
}

inline void cudaErrCheck(const cudaError_t res)
{
    if (res != cudaSuccess)
        ERR_AND_DIE(cudaGetErrorString(res));
}

#endif //ITS_RAINING_SPHERES_COMMON_CUH