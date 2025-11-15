#ifndef CUDA_RAYCAST_OBJ_PARSER_CUH
#define CUDA_RAYCAST_OBJ_PARSER_CUH

#include "common.cuh"
#include "calc.cuh"

// parser "not-quite-.obj" files
// supports the following types of entries:
// - comments (beginning with a `#`)
// - v x y z (vertex)
// - vn x y z (normal)
// - f v1/vn1 v2/vn2 v3/vn3
// - color r g b (values 0-255, the color of the object)
// - ks k_s (specular reflection coeff)
// - kd k_d (diffuse reflection coeff)
// - ka k_a (ambient reflection coeff)
// - alpha alpha (shininess constant)
// - light x y z r g b (an isotropic light source with color rgb at (x, y, z))
class ObjParser
{
public:
    std::vector<v3> vertices;
    std::vector<v3> normals;
    std::vector<Triangle> faces;

    void parseFile(const char* path);

private:
    static v3 parseVertex(const std::string& data);

    static v3 parseNormal(const std::string& data);

    Triangle parseFace(const std::string& data);

    void parseLine(const std::string& line);
};

#endif //CUDA_RAYCAST_OBJ_PARSER_CUH