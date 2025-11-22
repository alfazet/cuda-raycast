#ifndef CUDA_RAYCAST_OBJ_PARSER_CUH
#define CUDA_RAYCAST_OBJ_PARSER_CUH

#include "common.cuh"
#include "calc.cuh"
#include "light.cuh"

// parses simplified .obj files
// supports the following types of entries:
// - v x y z (vertex)
// - vn x y z (normal)
// - f v1/vn1 v2/vn2 v3/vn3 (or f v1 v2 v3 if no normals were defined)
// invalid entries are ignored
class ObjParser
{
public:
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<Triangle> faces;
    std::vector<uint3> facesIndices;
    // ordered by the faces they correspond to
    std::vector<Normals> orderedNormals;
    // neighbors[i] = list of faces that are neighbors of vertex i
    std::vector<std::vector<Triangle> > neighbors;

    void parseFile(const char* path);

private:
    float3 parseVertex(const std::string& data);

    float3 parseNormal(const std::string& data);

    Triangle parseFace(const std::string& data);

    std::tuple<Triangle, Normals> parseFaceWithNormals(const std::string& data);

    void parseLine(const std::string& line);
};

#endif //CUDA_RAYCAST_OBJ_PARSER_CUH