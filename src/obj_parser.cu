#include "obj_parser.cuh"

float3 ObjParser::parseVertex(const std::string& data)
{
    std::stringstream ss(data);
    float x, y, z;
    ss << x << y << z;

    return float3(x, y, z);
}

float2 ObjParser::parseTexture(const std::string& data)
{
    std::stringstream ss(data);
    float u, v;
    ss << u << v;

    return float2(u, v);
}

glm::vec3 ObjParser::parseNormal(const std::string& data)
{
    std::stringstream ss(data);
    float x, y, z;
    ss << x << y << z;

    return glm::vec3(x, y, z);
}

Triangle ObjParser::parseFace(const std::string& data)
{
    std::stringstream ss(data);
    std::string a, b, c;
    ss << a << b << c;

    int va, vb, vc, vta, vtb, vtc, vna, vnb, vnc;
    // split on slashes
}

void ObjParser::parseLine(const std::string& line)
{
    int firstSpace = line.find(' ');
    if (firstSpace == std::string::npos)
    {
        firstSpace = line.length();
    }
    std::string_view firstToken = line.substr(0, firstSpace);
    if (firstToken == "#")
    {
        std::cerr << "a comment\n";
    }
    else if (firstToken == "v")
    {
        this->parseVertex(line.substr(firstSpace + 1));
    }
    else if (firstToken == "vt")
    {
        this->parseTexture(line.substr(firstSpace + 1));
    }
    else if (firstToken == "vn")
    {
        this->parseNormal(line.substr(firstSpace + 1));
    }
    else if (firstToken == "f")
    {
        this->parseFace(line.substr(firstSpace + 1));
    }
    else
    {
        std::cerr << "something invalid\n";
    }
}

void ObjParser::parse(const char* path)
{
    std::string slurped = slurpFile(path), line;
    std::stringstream ss(slurped);
    while (std::getline(ss, line, '\n'))
    {
        this->parseLine(line);
    }
}
