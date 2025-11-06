#ifndef CUDA_RAYCAST_SHADER_CUH
#define CUDA_RAYCAST_SHADER_CUH

#include "common.cuh"

class Shader
{
public:
    uint programId;

    Shader(const char* vert_shader_path, const char* frag_shader_path);
};

#endif //CUDA_RAYCAST_SHADER_CUH