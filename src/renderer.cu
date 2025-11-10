#include "renderer.cuh"

__global__ void shadingKernel(uchar3* texBuf, int width, int height)
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

    if (x < 1.5)
    {
        texBuf[ty * width + tx] = uchar3(255, 0, 0);
    }
    else if (x < 1.77)
    {
        texBuf[ty * width + tx] = uchar3(0, 0, 255);
    }
}

Renderer::Renderer(uint pbo, int width, int height) : m_width{width}, m_height{height}, m_texBuf{nullptr},
                                                      m_blockDim{CUDA_BLOCK_DIM}
{
    this->m_gridDim = dim3((this->m_width + this->m_blockDim.x - 1) / this->m_blockDim.x,
                           (this->m_height + this->m_blockDim.y - 1) / this->m_blockDim.y, 1);
    cudaErrCheck(cudaGraphicsGLRegisterBuffer(&this->m_pboRes, pbo, cudaGraphicsMapFlagsWriteDiscard));
    cudaErrCheck(cudaMalloc(&this->m_texBuf, this->m_width * this->m_height * sizeof(uchar3)));
}

Renderer::~Renderer()
{
    cudaErrCheck(cudaGraphicsUnregisterResource(this->m_pboRes));
    cudaErrCheck(cudaFree(this->m_texBuf));
}

void Renderer::render(std::vector<Triangle>& faces)
{
    int nFaces = faces.size();
    Triangle* hFaces;
    cudaErrCheck(cudaMalloc(reinterpret_cast<void**>(&hFaces), nFaces * sizeof(Triangle)));
    cudaErrCheck(cudaMemcpy(hFaces, faces.data(), nFaces * sizeof(Triangle), cudaMemcpyHostToDevice));

    cudaErrCheck(cudaGraphicsMapResources(1, &this->m_pboRes));
    cudaErrCheck(cudaGraphicsResourceGetMappedPointer(&this->m_texBuf, nullptr, this->m_pboRes));
    shadingKernel<<<this->m_gridDim, this->m_blockDim>>>(static_cast<uchar3*>(this->m_texBuf), this->m_width,
                                                         this->m_height);
    cudaErrCheck(cudaGraphicsUnmapResources(1, &this->m_pboRes));

    cudaErrCheck(cudaFree(hFaces));
}