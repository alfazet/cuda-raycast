#ifndef CUDA_RAYCAST_LIGHT_PARSER_CUH
#define CUDA_RAYCAST_LIGHT_PARSER_CUH

#include "common.cuh"
#include "calc.cuh"
#include "light.cuh"

// parses files describing lighting properties and light sources
// supports the following types of entries:
// - color r g b (floats 0.0-1.0, the color of the object)
// - kd k_d (diffuse reflection coeff)
// - ks k_s (specular reflection coeff)
// - ka k_a (ambient reflection coeff)
// - alpha alpha (shininess constant)
// - light x y z r g b (a light source with color rgb at (x, y, z), rgb values are floats 0.0-1.0)
// invalid entries are ignored
class LightParser
{
public:
    std::vector<Light> lights;
    float3 objectColor;
    float kD, kS, kA, alpha;

    void parseFile(const char* path);

private:
    Light parseLight(const std::string& data) const;

    float parseFloat(const std::string& data) const;

    float3 parseColor(const std::string& data) const;

    void parseLine(const std::string& line);
};

#endif //CUDA_RAYCAST_LIGHT_PARSER_CUH