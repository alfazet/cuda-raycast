#include "renderer.cuh"

#include "app.cuh"

__global__ void shadingKernel(uchar3* texBuf, int width, int height)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= width || ty >= height)
    {
        return;
    }
    float rx = tx / static_cast<float>(width);
    float ry = ty / static_cast<float>(height);
    texBuf[ty * width + tx] = uchar3(static_cast<unsigned char>(255.0 * rx), 0, static_cast<unsigned char>(255.0 * ry));
}

Renderer::Renderer(uint pbo, int width, int height) : m_width{width}, m_height{height}, m_blockDim{CUDA_BLOCK_DIM},
                                                      m_texBuf{nullptr}
{
    this->m_gridDim = dim3((this->m_width + this->m_blockDim.x - 1) / this->m_blockDim.x,
                           (this->m_height + this->m_blockDim.y - 1) / this->m_blockDim.y, 1);
    cudaErrCheck(cudaGraphicsGLRegisterBuffer(&this->m_pboRes, pbo, cudaGraphicsMapFlagsWriteDiscard));
    cudaErrCheck(cudaMalloc(&this->m_texBuf, this->m_width * this->m_height * sizeof(uchar3)));
}

Renderer::~Renderer()
{
    cudaErrCheck(cudaGraphicsUnregisterResource(this->m_pboRes));
    // this is broken for some reason
    cudaErrCheck(cudaFree(this->m_texBuf));
}

void Renderer::render()
{
    cudaErrCheck(cudaGraphicsMapResources(1, &this->m_pboRes));
    cudaErrCheck(cudaGraphicsResourceGetMappedPointer(&this->m_texBuf, nullptr, this->m_pboRes));
    shadingKernel<<<this->m_gridDim, this->m_blockDim>>>(static_cast<uchar3*>(this->m_texBuf), this->m_width,
                                                         this->m_height);
    cudaErrCheck(cudaGraphicsUnmapResources(1, &this->m_pboRes));
}