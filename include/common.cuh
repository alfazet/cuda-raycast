#ifndef CUDA_RAYCAST_COMMON_CUH
#define CUDA_RAYCAST_COMMON_CUH

#include <iostream>
#include <format>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "glad/gl.h"
#define GLFW_INCLUDE_GLU 1
#include "GLFW/glfw3.h"

#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "utils.cuh"

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

inline std::string slurpFile(const char* file_path)
{
    std::ifstream file_stream(file_path);
    if (file_stream.fail())
    {
        ERR_AND_DIE(std::format("can't open file `{}`", file_path));
    }
    std::stringstream content;
    content << file_stream.rdbuf();
    file_stream.close();

    return content.str();
}

#endif //CUDA_RAYCAST_COMMON_CUH
