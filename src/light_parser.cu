#include "light_parser.cuh"

Light LightParser::parseLight(const std::string& data) const
{
    std::stringstream ss(data);
    float x, y, z, r, g, b;
    ss >> x >> y >> z >> r >> g >> b;

    return Light{
        .pos = make_float3(x, y, z),
        .color = make_float3(r, g, b),
    };
}

float LightParser::parseFloat(const std::string& data) const
{
    std::stringstream ss(data);
    float x;
    ss >> x;

    return x;
}

float3 LightParser::parseColor(const std::string& data) const
{
    std::stringstream ss(data);
    float r, g, b;
    ss >> r >> g >> b;

    return make_float3(r, g, b);
}

void LightParser::parseLine(const std::string& line)
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
    if (firstToken == "color")
    {
        this->objectColor = this->parseColor(rest);
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

void LightParser::parseFile(const char* path)
{
    std::string slurped = slurpFile(path), line;
    std::stringstream ss(slurped);
    while (std::getline(ss, line, '\n'))
    {
        this->parseLine(line);
    }
}