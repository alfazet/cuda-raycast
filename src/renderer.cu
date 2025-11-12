#include "renderer.cuh"

__global__ void transRotKernel(Triangle* faces, Triangle* facesOriginal, int nFaces, v3 shift, float scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx >= nFaces)
    {
        return;
    }
    faces[tx].a = scale * (facesOriginal[tx].a + shift);
    faces[tx].b = scale * (facesOriginal[tx].b + shift);
    faces[tx].c = scale * (facesOriginal[tx].c + shift);
}

__global__ void shadingKernel(uchar3* texBuf, int width, int height, Triangle* faces, int nFaces)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= width || ty >= height)
    {
        return;
    }
    float aspect_ratio = width / static_cast<float>(height);
    float x = 2.0f * tx / static_cast<float>(width) * aspect_ratio - 1.0f;
    float y = 2.0f * ty / static_cast<float>(height) - 1.0f;
    texBuf[ty * width + tx] = uchar3(0, 0, 0);

    // TODO: don't hardcode this
    v3 origin = v3(x, y, 4.0f), rayDir = v3(0.0f, 0.0f, -1.0f);
    int hitIdx = -1;
    float minT = FLT_MAX; // for depth-buffering
    for (int i = 0; i < nFaces; i++)
    {
        float t = calc::triangleIntersection(origin, rayDir, faces[i]);
        if (t > 0.0 && t < minT)
        {
            minT = t;
            hitIdx = i;
        }
    }

    switch (hitIdx)
    {
    case 0:
        texBuf[ty * width + tx] = uchar3(255, 0, 0);
        break;
    case 1:
        texBuf[ty * width + tx] = uchar3(0, 255, 0);
        break;
    case 2:
        texBuf[ty * width + tx] = uchar3(0, 0, 255);
        break;
    case 3:
        texBuf[ty * width + tx] = uchar3(128, 0, 128);
        break;
    case 4:
    case 5:
        texBuf[ty * width + tx] = uchar3(0, 128, 128);
        break;
    default:
        texBuf[ty * width + tx] = uchar3(128, 128, 128);
    }
}

Renderer::Renderer(uint pbo, int width, int height, std::vector<Triangle>& faces) : m_width{width}, m_height{height},
    m_nFaces{static_cast<int>(faces.size())}, m_blockDim{CUDA_BLOCK_DIM}
{
    this->m_gridDim = dim3((this->m_width + this->m_blockDim.x - 1) / this->m_blockDim.x,
                           (this->m_height + this->m_blockDim.y - 1) / this->m_blockDim.y, 1);
    this->camera = new Camera(this->m_width, this->m_height, M_PI_2, 0.01f, 100.0f);
    cudaErrCheck(cudaGraphicsGLRegisterBuffer(&this->m_pboRes, pbo, cudaGraphicsMapFlagsWriteDiscard));
    cudaErrCheck(cudaMalloc(&this->m_dTexBuf, this->m_width * this->m_height * sizeof(uchar3)));
    cudaErrCheck(cudaMalloc(&this->m_dFaces, this->m_nFaces * sizeof(Triangle)));
    cudaErrCheck(cudaMalloc(&this->m_dFacesOriginal, this->m_nFaces * sizeof(Triangle)));
    cudaErrCheck(cudaMemcpy(this->m_dFacesOriginal, faces.data(), this->m_nFaces * sizeof(Triangle),
                            cudaMemcpyHostToDevice));
}

Renderer::~Renderer()
{
    cudaErrCheck(cudaGraphicsUnregisterResource(this->m_pboRes));
    cudaErrCheck(cudaFree(this->m_dFacesOriginal));
    cudaErrCheck(cudaFree(this->m_dTexBuf));
    cudaErrCheck(cudaFree(this->m_dFaces));
    delete this->camera;
}

void Renderer::render()
{
    cudaErrCheck(cudaGraphicsMapResources(1, &this->m_pboRes));
    cudaErrCheck(cudaGraphicsResourceGetMappedPointer(&this->m_dTexBuf, nullptr, this->m_pboRes));
    transRotKernel<<<this->m_gridDim, this->m_blockDim>>>(this->m_dFaces, this->m_dFacesOriginal, this->m_nFaces,
                                                          this->camera->shift, this->camera->scale);
    shadingKernel<<<this->m_gridDim, this->m_blockDim>>>(static_cast<uchar3*>(this->m_dTexBuf), this->m_width,
                                                         this->m_height, this->m_dFaces, this->m_nFaces);
    cudaErrCheck(cudaGraphicsUnmapResources(1, &this->m_pboRes));
}

void Renderer::handleKey(int key, float dt)
{
    this->camera->handleKey(key, dt);
}

void Renderer::handleMouse(v2 delta)
{
    this->camera->handleMouse(delta);
}
