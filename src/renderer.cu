#include "renderer.cuh"

__global__ void shadingKernel(uchar3* texBuf, int width, int height, Triangle* faces, int nFaces, Light* lights,
                              int nLights, Camera camera)
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
    v3 rayDir = glm::normalize(m3(camera.right, camera.up, camera.forward) * v3(x, y, 1.0f));

    int hitIdx = -1;
    float minT = FLT_MAX; // for depth-buffering
    for (int i = 0; i < nFaces; i++)
    {
        float t = calc::triangleIntersection(camera.pos, rayDir, faces[i]);
        if (t > 0.0 && t < minT)
        {
            minT = t;
            hitIdx = i;
        }
    }
    if (hitIdx == -1)
    {
        texBuf[ty * width + tx] = uchar3(0, 0, 0);
        return;
    }

    v3 hitPoint = camera.pos + minT * rayDir;
    v3 normal = calc::normalAt(hitPoint, faces[hitIdx]);
    float3 color(0.0f, 0.0f, 0.0f), surfaceColor(1.0f, 1.0f, 1.0f);
    float kd = 0.75, ks = 0.5, alpha = 3;
    for (int i = 0; i < nLights; i++)
    {
        v3 toLight = glm::normalize(lights[i].pos - hitPoint);
        float nlDot = max(0.0f, glm::dot(normal, toLight));
        v3 view = glm::normalize(hitPoint - camera.pos);
        v3 rVec = glm::normalize(2.0f * nlDot * normal - toLight);
        float rvDot = max(0.0f, glm::dot(rVec, view));

        float r = min(lights[i].color.x * surfaceColor.x * (nlDot * kd + rvDot * ks), 1.0f);
        float g = min(lights[i].color.y * surfaceColor.y * (nlDot * kd + rvDot * ks), 1.0f);
        float b = min(lights[i].color.z * surfaceColor.z * (nlDot * kd + rvDot * ks), 1.0f);
        color.x += r;
        color.y += g;
        color.z += b;
    }
    texBuf[ty * width + tx] = calc::colorBytes(color);
}

Renderer::Renderer(uint pbo, int width, int height, std::vector<Triangle>& faces,
                   std::vector<Light>& lights) : m_width{width}, m_height{height},
                                                 m_nFaces{static_cast<int>(faces.size())},
                                                 m_nLights{static_cast<int>(lights.size())}, m_blockDim{CUDA_BLOCK_DIM}
{
    this->m_gridDim = dim3((this->m_width + this->m_blockDim.x - 1) / this->m_blockDim.x,
                           (this->m_height + this->m_blockDim.y - 1) / this->m_blockDim.y, 1);
    this->camera = Camera(v3(0.0f, 0.0f, 5.0f), 0.0f, 0.0f, M_PI_2);
    cudaErrCheck(cudaGraphicsGLRegisterBuffer(&this->m_pboRes, pbo, cudaGraphicsMapFlagsWriteDiscard));
    cudaErrCheck(cudaMalloc(&this->m_dTexBuf, this->m_width * this->m_height * sizeof(uchar3)));
    cudaErrCheck(cudaMalloc(&this->m_dFaces, this->m_nFaces * sizeof(Triangle)));
    cudaErrCheck(cudaMemcpy(this->m_dFaces, faces.data(), this->m_nFaces * sizeof(Triangle), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMalloc(&this->m_dLights, this->m_nLights * sizeof(Light)));
    cudaErrCheck(cudaMemcpy(this->m_dLights, lights.data(), this->m_nLights * sizeof(Light), cudaMemcpyHostToDevice));
}

Renderer::~Renderer()
{
    cudaErrCheck(cudaGraphicsUnregisterResource(this->m_pboRes));
    cudaErrCheck(cudaFree(this->m_dTexBuf));
    cudaErrCheck(cudaFree(this->m_dFaces));
}

void Renderer::render()
{
    cudaErrCheck(cudaGraphicsMapResources(1, &this->m_pboRes));
    cudaErrCheck(cudaGraphicsResourceGetMappedPointer(&this->m_dTexBuf, nullptr, this->m_pboRes));
    shadingKernel<<<this->m_gridDim, this->m_blockDim>>>(static_cast<uchar3*>(this->m_dTexBuf), this->m_width,
                                                         this->m_height, this->m_dFaces, this->m_nFaces,
                                                         this->m_dLights, this->m_nLights, this->camera);
    cudaErrCheck(cudaGraphicsUnmapResources(1, &this->m_pboRes));
}

void Renderer::handleKey(int key, float dt)
{
    this->camera.handleKey(key, dt);
}
