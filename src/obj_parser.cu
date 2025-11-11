#include "obj_parser.cuh"

v3 ObjParser::parseVertex(const std::string& data)
{
    std::stringstream ss(data);
    float x, y, z;
    ss >> x >> y >> z;

    return {x, y, z};
}

v2 ObjParser::parseTexture(const std::string& data)
{
    std::stringstream ss(data);
    float u, v;
    ss >> u >> v;
    assert(u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0);

    return {u, v};
}

v3 ObjParser::parseNormal(const std::string& data)
{
    std::stringstream ss(data);
    float x, y, z;
    ss >> x >> y >> z;
    v3 v(x, y, z);
    v = normalize(v);

    return v;
}

Triangle ObjParser::parseFace(const std::string& data)
{
    std::stringstream ss(data);
    std::string a, b, c;
    ss >> a >> b >> c;
    std::stringstream fa(a), fb(b), fc(c);
    std::string s;
    int va, vb, vc, vta, vtb, vtc, vna, vnb, vnc;

    std::getline(fa, s, '/');
    va = std::stoi(s);
    std::getline(fa, s, '/');
    vta = std::stoi(s);
    std::getline(fa, s, ' ');
    vna = std::stoi(s);

    std::getline(fb, s, '/');
    vb = std::stoi(s);
    std::getline(fb, s, '/');
    vtb = std::stoi(s);
    std::getline(fb, s, ' ');
    vnb = std::stoi(s);

    std::getline(fc, s, '/');
    vc = std::stoi(s);
    std::getline(fc, s, '/');
    vtc = std::stoi(s);
    std::getline(fc, s, ' ');
    vnc = std::stoi(s);

    return Triangle{
        .a = this->vertices[va - 1],
        .b = this->vertices[vb - 1],
        .c = this->vertices[vc - 1],
        .uva = this->texVertices[vta - 1],
        .uvb = this->texVertices[vtb - 1],
        .uvc = this->texVertices[vtc - 1],
        .na = this->normals[vna - 1],
        .nb = this->normals[vnb - 1],
        .nc = this->normals[vnc - 1],
    };
}

void ObjParser::parseLine(const std::string& line)
{
    int firstSpace = line.find(' ');
    if (firstSpace == std::string::npos)
    {
        firstSpace = line.length();
    }
    std::string firstToken = line.substr(0, firstSpace);
    if (firstToken == "v")
    {
        this->vertices.emplace_back(this->parseVertex(line.substr(firstSpace + 1)));
    }
    else if (firstToken == "vt")
    {
        this->texVertices.emplace_back(this->parseTexture(line.substr(firstSpace + 1)));
    }
    else if (firstToken == "vn")
    {
        this->normals.emplace_back(this->parseNormal(line.substr(firstSpace + 1)));
    }
    else if (firstToken == "f")
    {
        this->faces.emplace_back(this->parseFace(line.substr(firstSpace + 1)));
    }
}

void ObjParser::parseFile(const char* path)
{
    std::string slurped = slurpFile(path), line;
    std::stringstream ss(slurped);
    while (std::getline(ss, line, '\n'))
    {
        this->parseLine(line);
    }
}