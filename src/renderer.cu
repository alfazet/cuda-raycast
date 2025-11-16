#include "renderer.cuh"

__global__ void shadingKernel(uchar3* texBuf, int width, int height, Triangle* faces, int nFaces, Light* lights,
                              int nLights, Normals* normals, float3 surfaceColor, float kD, float kS, float kA,
                              float alpha, Camera camera)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= width || ty >= height)
    {
        return;
    }
    float aspect_ratio = width / static_cast<float>(height);
    float x = (2.0f * (tx / static_cast<float>(width)) - 1.0f) * aspect_ratio * tanf(0.5f * camera.fov);
    float y = (2.0f * (ty / static_cast<float>(height)) - 1.0f) * tanf(0.5f * camera.fov);
    float3 rayDir = make_float3(
        x * camera.right.x + y * camera.up.x + camera.forward.x,
        x * camera.right.y + y * camera.up.y + camera.forward.y,
        x * camera.right.z + y * camera.up.z + camera.forward.z);
    float3 cameraPos = make_float3(camera.pos.x, camera.pos.y, camera.pos.z);

    int hitIdx = -1;
    float t, minT = FLT_MAX; // for depth-buffering
    float3 hitPoint, hitPointBary, normal;
    for (int i = 0; i < nFaces; i++)
    {
        triangleIntersection(cameraPos, rayDir, &faces[i], &t, &hitPointBary);
        if (t > 0.0 && t < minT)
        {
            hitPoint = cameraPos + t * rayDir;
            minT = t;
            hitIdx = i;
            normal = normalize(
                hitPointBary.x * normals[i].na + hitPointBary.y * normals[i].nb + hitPointBary.z * normals[i].nc);
        }
    }
    if (hitIdx == -1)
    {
        texBuf[ty * width + tx] = uchar3(128, 128, 128);
        return;
    }

    float3 resColor(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < nLights; i++)
    {
        float3 toLight = normalize(lights[i].pos - hitPoint);
        float nlDot = max(dot(normal, toLight), 0.0f);
        float3 view = normalize(cameraPos - hitPoint);
        float3 rVec = normalize(2.0f * nlDot * normal - toLight);
        float rvDot = max(dot(rVec, view), 0.0f);
        float rvDotPow = powf(rvDot, alpha);
        resColor += lights[i].color * surfaceColor * (nlDot * kD + rvDotPow * kS);
    }
    float ambientI = min(0.25f * nLights, 1.0f);
    resColor += kA * ambientI * surfaceColor;
    resColor = clamp(resColor, ZERO, ONE);
    texBuf[ty * width + tx] = rgbFloatsToBytes(resColor);
}

Renderer::Renderer(uint pbo, int width, int height, std::vector<Triangle>& faces, std::vector<Normals>& normals,
                   std::vector<Light>& lights, float3 color, float kD, float kS, float kA,
                   float alpha) : m_width{width}, m_height{height},
                                  m_nFaces{static_cast<int>(faces.size())},
                                  m_nLights{static_cast<int>(lights.size())}, m_kD{kD},
                                  m_kS{kS}, m_kA{kA}, m_alpha{alpha}, m_color{color}, m_blockDim{CUDA_BLOCK_DIM}
{
    this->m_gridDim = dim3((this->m_width + this->m_blockDim.x - 1) / this->m_blockDim.x,
                           (this->m_height + this->m_blockDim.y - 1) / this->m_blockDim.y, 1);
    this->camera = Camera(M_PI_2);
    cudaErrCheck(cudaGraphicsGLRegisterBuffer(&this->m_pboRes, pbo, cudaGraphicsMapFlagsWriteDiscard));
    cudaErrCheck(cudaMalloc(&this->m_dTexBuf, this->m_width * this->m_height * sizeof(uchar3)));
    cudaErrCheck(cudaMalloc(&this->m_dFaces, this->m_nFaces * sizeof(Triangle)));
    cudaErrCheck(cudaMemcpy(this->m_dFaces, faces.data(), this->m_nFaces * sizeof(Triangle), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMalloc(&this->m_dLights, this->m_nLights * sizeof(Light)));
    cudaErrCheck(cudaMemcpy(this->m_dLights, lights.data(), this->m_nLights * sizeof(Light), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMalloc(&this->m_dNormals, this->m_nFaces * sizeof(Normals)));
    cudaErrCheck(cudaMemcpy(this->m_dNormals, normals.data(), this->m_nFaces * sizeof(Normals),
                            cudaMemcpyHostToDevice));
}

Renderer::~Renderer()
{
    cudaErrCheck(cudaGraphicsUnregisterResource(this->m_pboRes));
    cudaErrCheck(cudaFree(this->m_dNormals));
    cudaErrCheck(cudaFree(this->m_dLights));
    cudaErrCheck(cudaFree(this->m_dFaces));
    cudaErrCheck(cudaFree(this->m_dTexBuf));
}

void Renderer::render()
{
    cudaErrCheck(cudaGraphicsMapResources(1, &this->m_pboRes));
    cudaErrCheck(cudaGraphicsResourceGetMappedPointer(&this->m_dTexBuf, nullptr, this->m_pboRes));
    shadingKernel<<<this->m_gridDim, this->m_blockDim>>>(static_cast<uchar3*>(this->m_dTexBuf), this->m_width,
                                                         this->m_height, this->m_dFaces, this->m_nFaces,
                                                         this->m_dLights, this->m_nLights, this->m_dNormals,
                                                         this->m_color, this->m_kD,
                                                         this->m_kS, this->m_kA, this->m_alpha, this->camera);
    cudaErrCheck(cudaGraphicsUnmapResources(1, &this->m_pboRes));
}

void Renderer::handleKey(int key, float dt)
{
    this->camera.handleKey(key, dt);
}