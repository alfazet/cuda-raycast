#include "obj_parser.cuh"

float3 ObjParser::parseVertex(const std::string& data) const
{
    std::stringstream ss(data);
    float x, y, z;
    ss >> x >> y >> z;

    return make_float3(x, y, z);
}

float3 ObjParser::parseNormal(const std::string& data) const
{
    std::stringstream ss(data);
    float x, y, z;
    ss >> x >> y >> z;

    return normalize(make_float3(x, y, z));
}

Triangle ObjParser::parseFace(const std::string& data) const
{
    std::stringstream ss(data);
    int va, vb, vc;
    ss >> va >> vb >> vc;

    return Triangle{
        .a = this->vertices[va - 1],
        .b = this->vertices[vb - 1],
        .c = this->vertices[vc - 1],
    };
}

std::tuple<Triangle, Normals> ObjParser::parseFaceWithNormals(const std::string& data) const
{
    std::stringstream ss(data);
    std::string a, b, c;
    ss >> a >> b >> c;
    std::stringstream fa(a), fb(b), fc(c);
    std::string s;
    int va, vb, vc, vna, vnb, vnc;

    std::getline(fa, s, '/');
    va = std::stoi(s);
    std::getline(fa, s, ' ');
    vna = std::stoi(s);

    std::getline(fb, s, '/');
    vb = std::stoi(s);
    std::getline(fb, s, ' ');
    vnb = std::stoi(s);

    std::getline(fc, s, '/');
    vc = std::stoi(s);
    std::getline(fc, s, ' ');
    vnc = std::stoi(s);

    auto triangle = Triangle{
        .a = this->vertices[va - 1],
        .b = this->vertices[vb - 1],
        .c = this->vertices[vc - 1],
    };
    auto normals = Normals{
        .na = this->normals[vna - 1],
        .nb = this->normals[vnb - 1],
        .nc = this->normals[vnc - 1],
    };

    return {triangle, normals};
}

Light ObjParser::parseLight(const std::string& data) const
{
    std::stringstream ss(data);
    float x, y, z, r, g, b;
    ss >> x >> y >> z >> r >> g >> b;

    return Light{
        .pos = make_float3(x, y, z),
        .color = make_float3(r, g, b),
    };
}

float ObjParser::parseFloat(const std::string& data) const
{
    std::stringstream ss(data);
    float x;
    ss >> x;

    return x;
}

float3 ObjParser::parseColor(const std::string& data) const
{
    std::stringstream ss(data);
    float r, g, b;
    ss >> r >> g >> b;

    return make_float3(r, g, b);
}

void ObjParser::parseLine(const std::string& line)
{
    if (line.empty())
    {
        return;
    }
    size_t firstSpace = line.find(' ');
    if (firstSpace == std::string::npos)
    {
        firstSpace = line.length();
    }
    std::string firstToken = line.substr(0, firstSpace);
    std::string rest = line.substr(firstSpace + 1);
    if (firstToken == "v")
    {
        this->vertices.emplace_back(this->parseVertex(rest));
    }
    else if (firstToken == "vn")
    {
        this->normals.emplace_back(this->parseNormal(rest));
    }
    else if (firstToken == "f")
    {
        if (this->normals.empty())
        {
            // obj file without normals
            this->faces.emplace_back(this->parseFace(rest));
        }
        else
        {
            auto [triangle, normals] = this->parseFaceWithNormals(rest);
            this->faces.push_back(triangle);
            this->orderedNormals.push_back(normals);
        }
    }
    else if (firstToken == "color")
    {
        this->color = this->parseColor(rest);
    }
    else if (firstToken == "light")
    {
        this->lights.push_back(this->parseLight(rest));
    }
    else if (firstToken == "kd")
    {
        this->kD = this->parseFloat(rest);
    }
    else if (firstToken == "ks")
    {
        this->kS = this->parseFloat(rest);
    }
    else if (firstToken == "ka")
    {
        this->kA = this->parseFloat(rest);
    }
    else if (firstToken == "alpha")
    {
        this->alpha = this->parseFloat(rest);
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