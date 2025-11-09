#include "shader.cuh"

constexpr int INFO_LOG_LEN = 2048;

void checkShaderCompilation(uint shader)
{
    int ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok == GL_FALSE)
    {
        char info_log[INFO_LOG_LEN];
        glGetShaderInfoLog(shader, INFO_LOG_LEN, nullptr, info_log);
        ERR_AND_DIE(std::string("shader compilation failed (") + info_log + ")");
    }
}

void checkProgramLinking(uint program)
{
    int ok;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (ok == GL_FALSE)
    {
        char info_log[INFO_LOG_LEN];
        glGetProgramInfoLog(program, INFO_LOG_LEN, nullptr, info_log);
        ERR_AND_DIE(std::string("shader program linking failed (") + info_log + ")");
    }
}

Shader::Shader(const char* vert_shader_path, const char* frag_shader_path)
{
    std::string vert_shader_code_ = slurpFile(vert_shader_path);
    std::string frag_shader_code_ = slurpFile(frag_shader_path);
    const char* vert_shader_code = vert_shader_code_.c_str();
    const char* frag_shader_code = frag_shader_code_.c_str();
    uint vert_shader = glCreateShader(GL_VERTEX_SHADER);
    uint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    uint programId = glCreateProgram();

    glShaderSource(vert_shader, 1, &vert_shader_code, nullptr);
    glCompileShader(vert_shader);
    checkShaderCompilation(vert_shader);
    glShaderSource(frag_shader, 1, &frag_shader_code, nullptr);
    glCompileShader(frag_shader);
    checkShaderCompilation(frag_shader);

    glAttachShader(programId, vert_shader);
    glAttachShader(programId, frag_shader);
    glLinkProgram(programId);
    checkProgramLinking(programId);

    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);

    this->programId = programId;
}