#include "renderer.cuh"

__global__ void shadingKernel(uchar3* texBuf, int width, int height, Triangle* faces, int nFaces)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= width || ty >= height)
    {
        return;
    }
    float aspect_ratio = width / static_cast<float>(height);
    float x = tx / static_cast<float>(width) * aspect_ratio;
    float y = ty / static_cast<float>(height);
    texBuf[ty * width + tx] = uchar3(0, 0, 0);

    int hitIdx = -1;
    v3 ray_dir = v3(0.0f, 0.0f, -1.0f);
    v3 origin = v3(x, y, 0.5f);
    float minT = FLT_MAX; // for depth-buffering
    for (int i = 0; i < nFaces; i++)
    {
        float t = calc::triangleIntersection(origin, ray_dir, faces[i]);
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
    }
}

Renderer::Renderer(uint pbo, int width, int height, std::vector<Triangle>& faces) : m_width{width}, m_height{height},
    m_nFaces{static_cast<int>(faces.size())}, m_blockDim{CUDA_BLOCK_DIM}
{
    this->m_gridDim = dim3((this->m_width + this->m_blockDim.x - 1) / this->m_blockDim.x,
                           (this->m_height + this->m_blockDim.y - 1) / this->m_blockDim.y, 1);
    cudaErrCheck(cudaGraphicsGLRegisterBuffer(&this->m_pboRes, pbo, cudaGraphicsMapFlagsWriteDiscard));
    cudaErrCheck(cudaMalloc(&this->m_dTexBuf, this->m_width * this->m_height * sizeof(uchar3)));
    cudaErrCheck(cudaMalloc(&this->m_dFaces, this->m_nFaces * sizeof(Triangle)));
    cudaErrCheck(cudaMemcpy(this->m_dFaces, faces.data(), this->m_nFaces * sizeof(Triangle), cudaMemcpyHostToDevice));
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
                                                         this->m_height, this->m_dFaces, this->m_nFaces);
    cudaErrCheck(cudaGraphicsUnmapResources(1, &this->m_pboRes));
}