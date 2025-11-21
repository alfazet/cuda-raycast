#include "renderer.cuh"

__device__ void checkTriangleIntersections(float3 origin, float3 dir, Triangle* faces, int* idxs, int nFaces,
                                           float* minT, float3* minBary, int* hitIdx)
{
    float t;
    float3 bary;
    for (int i = 0; i < nFaces; i++)
    {
        if (idxs[i] == -1)
        {
            continue;
        }
        triangleIntersection(origin, dir, &faces[i], &t, &bary);
        if (t > 0.0f && t < *minT)
        {
            *minT = t;
            *minBary = bary;
            *hitIdx = idxs[i];
        }
    }
}

__device__ uchar3 computeColor(float3 hitPoint, float3 normal, float3 cameraPos, Light* lights, int nLights,
                               float3 surfaceColor, float kD, float kS, float kA,
                               float alpha)
{
    float3 resColor = ZERO;
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

    return rgbFloatsToBytes(resColor);
}

// TODO: SOA
template <int N> __global__ void shadingKernel(uchar3* texBuf, int width, int height, Triangle* faces, Normals* normals,
                                               int nFaces,
                                               Light* lights, int nLights, float3 surfaceColor, float kD, float kS,
                                               float kA, float alpha, float3 baseCameraPos, float3 rayDir)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tidxInBlock = threadIdx.y * blockDim.x + threadIdx.x;
    float aspect_ratio = width / static_cast<float>(height);
    float x = (2.0f * (tx / static_cast<float>(width)) - 1.0f) * aspect_ratio;
    float y = 2.0f * (ty / static_cast<float>(height)) - 1.0f;
    float3 cameraPos = baseCameraPos + make_float3(x, y, 0.0f);
    float xPerBlock = 2.0f * aspect_ratio / static_cast<float>(gridDim.x);
    float yPerBlock = 2.0f / static_cast<float>(gridDim.y);
    float blockMinX = baseCameraPos.x - aspect_ratio + blockIdx.x * xPerBlock, blockMinY =
        baseCameraPos.y - 1.0f + blockIdx.y * yPerBlock;
    // make blocks overlap to prevent visual glitches
    float blockMaxX = blockMinX + 2.0f * xPerBlock, blockMaxY = blockMinY + 2.0f * yPerBlock;

    float minT = FLT_MAX;
    int hitIdx = -1;
    float3 hitPointBary;
    __shared__ Triangle reachableFaces[N];
    __shared__ int reachableIdxs[N];
    for (int i = 0; i < nFaces; i += N)
    {
        int faceIdx = i + tidxInBlock, isReachable = 0;
        if (faceIdx < nFaces)
        {
            float3 vA = faces[faceIdx].a, vB = faces[faceIdx].b, vC = faces[faceIdx].c;
            float minX = min(vA.x, min(vB.x, vC.x));
            float minY = min(vA.y, min(vB.y, vC.y));
            float maxX = max(vA.x, max(vB.x, vC.x));
            float maxY = max(vA.y, max(vB.y, vC.y));
            isReachable = minX <= blockMaxX && minY <= blockMaxY && maxX >= blockMinX && maxY >= blockMinY;
        }
        if (isReachable)
        {
            reachableFaces[tidxInBlock] = faces[faceIdx];
            reachableIdxs[tidxInBlock] = faceIdx;
        }
        else
        {
            reachableIdxs[tidxInBlock] = -1;
        }
        int anyReachable = __syncthreads_or(isReachable);
        if (anyReachable && tx < width && ty < height)
        {
            checkTriangleIntersections(cameraPos, rayDir, reachableFaces, reachableIdxs, N, &minT, &hitPointBary,
                                       &hitIdx);
        }
        __syncthreads();
    }

    if (tx >= width || ty >= height)
    {
        return;
    }
    if (hitIdx == -1)
    {
        texBuf[ty * width + tx] = BKG_COLOR;
    }
    else
    {
        float3 normal = normalize(
            hitPointBary.x * normals[hitIdx].na + hitPointBary.y * normals[hitIdx].nb + hitPointBary.z * normals[hitIdx]
            .nc);
        float3 hitPoint = cameraPos + minT * rayDir;
        texBuf[ty * width + tx] = computeColor(hitPoint, normal, cameraPos, lights, nLights, surfaceColor, kD, kS, kA,
                                               alpha);
    }
}

/*
__global__ void shadingKernelSlow(uchar3* texBuf, int width, int height, Triangle* faces,
                                  Normals* normals,
                                  int nFaces,
                                  Light* lights, int nLights, float3 surfaceColor, float kD, float kS,
                                  float kA, float alpha, float3 baseCameraPos, float3 rayDir)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= width || ty >= height)
    {
        return;
    }
    float aspect_ratio = width / static_cast<float>(height);
    float x = (2.0f * (tx / static_cast<float>(width)) - 1.0f) * aspect_ratio;
    float y = 2.0f * (ty / static_cast<float>(height)) - 1.0f;
    float3 cameraPos = baseCameraPos + make_float3(x, y, 0.0f);

    float minT = FLT_MAX, t;
    int hitIdx = -1;
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
        texBuf[ty * width + tx] = BKG_COLOR;
    }
    else
    {
        texBuf[ty * width + tx] = computeColor(hitPoint, normal, cameraPos, lights, nLights, surfaceColor, kD, kS, kA,
                                               alpha);
    }
}
*/

// rotates and scales all faces (together with their normals)
__global__ void objectTransformationKernel(Triangle* originalFaces, Triangle* transFaces, Normals* originalNormals,
                                           Normals* rotatedNormals, int nFaces, float3 angles, float scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx >= nFaces)
    {
        return;
    }
    // rotation around X
    float3 row1x = make_float3(cosf(angles.x), -sinf(angles.x), 0.0f);
    float3 row2x = make_float3(sinf(angles.x), cosf(angles.x), 0.0f);
    float3 row3x = make_float3(0.0f, 0.0f, 1.0f);
    // rotation around Y
    float3 row1y = make_float3(1.0f, 0.0f, 0.0f);
    float3 row2y = make_float3(0.0f, cosf(angles.y), -sinf(angles.y));
    float3 row3y = make_float3(0.0f, sinf(angles.y), cosf(angles.y));
    // rotation around Z
    float3 row1z = make_float3(cosf(angles.z), -sinf(angles.z), 0.0f);
    float3 row2z = make_float3(sinf(angles.z), cosf(angles.z), 0.0f);
    float3 row3z = make_float3(0.0f, 0.0f, 1.0f);

    transFaces[tx].a = originalFaces[tx].a;
    transFaces[tx].b = originalFaces[tx].b;
    transFaces[tx].c = originalFaces[tx].c;
    rotatedNormals[tx].na = originalNormals[tx].na;
    rotatedNormals[tx].nb = originalNormals[tx].nb;
    rotatedNormals[tx].nc = originalNormals[tx].nc;

    transFaces[tx].a = vecMatMul3(transFaces[tx].a, row1x, row2x, row3x);
    transFaces[tx].a = vecMatMul3(transFaces[tx].a, row1y, row2y, row3y);
    transFaces[tx].a = vecMatMul3(transFaces[tx].a, row1z, row2z, row3z);
    transFaces[tx].b = vecMatMul3(transFaces[tx].b, row1x, row2x, row3x);
    transFaces[tx].b = vecMatMul3(transFaces[tx].b, row1y, row2y, row3y);
    transFaces[tx].b = vecMatMul3(transFaces[tx].b, row1z, row2z, row3z);
    transFaces[tx].c = vecMatMul3(transFaces[tx].c, row1x, row2x, row3x);
    transFaces[tx].c = vecMatMul3(transFaces[tx].c, row1y, row2y, row3y);
    transFaces[tx].c = vecMatMul3(transFaces[tx].c, row1z, row2z, row3z);

    transFaces[tx].a *= scale;
    transFaces[tx].b *= scale;
    transFaces[tx].c *= scale;

    rotatedNormals[tx].na = vecMatMul3(rotatedNormals[tx].na, row1x, row2x, row3x);
    rotatedNormals[tx].na = vecMatMul3(rotatedNormals[tx].na, row1y, row2y, row3y);
    rotatedNormals[tx].na = vecMatMul3(rotatedNormals[tx].na, row1z, row2z, row3z);
    rotatedNormals[tx].nb = vecMatMul3(rotatedNormals[tx].nb, row1x, row2x, row3x);
    rotatedNormals[tx].nb = vecMatMul3(rotatedNormals[tx].nb, row1y, row2y, row3y);
    rotatedNormals[tx].nb = vecMatMul3(rotatedNormals[tx].nb, row1z, row2z, row3z);
    rotatedNormals[tx].nc = vecMatMul3(rotatedNormals[tx].nc, row1x, row2x, row3x);
    rotatedNormals[tx].nc = vecMatMul3(rotatedNormals[tx].nc, row1y, row2y, row3y);
    rotatedNormals[tx].nc = vecMatMul3(rotatedNormals[tx].nc, row1z, row2z, row3z);
}

// rotates all lights by the specified angle around the Y axis
__global__ void lightRotationKernel(Light* originalLights, Light* rotatedLights, int nLights, float angle)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx >= nLights)
    {
        return;
    }
    float3 row1 = make_float3(cosf(angle), 0.0f, sinf(angle));
    float3 row2 = make_float3(0.0f, 1.0f, 0.0f);
    float3 row3 = make_float3(-sinf(angle), 0.0f, cosf(angle));
    rotatedLights[tx].pos = vecMatMul3(originalLights[tx].pos, row1, row2, row3);
}

Renderer::Renderer(uint pbo, int width, int height, std::vector<Triangle>& faces, std::vector<Normals>& normals,
                   std::vector<Light>& lights, float3 color, float kD, float kS, float kA,
                   float alpha) : m_width{width}, m_height{height},
                                  m_nFaces{static_cast<int>(faces.size())},
                                  m_nLights{static_cast<int>(lights.size())}, m_kD{kD},
                                  m_kS{kS}, m_kA{kA}, m_alpha{alpha}, m_color{color}
{
    cudaErrCheck(cudaGraphicsGLRegisterBuffer(&this->m_pboRes, pbo, cudaGraphicsMapFlagsWriteDiscard));
    cudaErrCheck(cudaMalloc(&this->m_dTexBuf, this->m_width * this->m_height * sizeof(uchar3)));

    cudaErrCheck(cudaMalloc(&this->m_dFaces, this->m_nFaces * sizeof(Triangle)));
    cudaErrCheck(cudaMemcpy(this->m_dFaces, faces.data(), this->m_nFaces * sizeof(Triangle), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMalloc(&this->m_dOriginalFaces, this->m_nFaces * sizeof(Triangle)));
    cudaErrCheck(cudaMemcpy(this->m_dOriginalFaces, this->m_dFaces, this->m_nFaces * sizeof(Triangle),
                            cudaMemcpyDeviceToDevice));

    cudaErrCheck(cudaMalloc(&this->m_dLights, this->m_nLights * sizeof(Light)));
    cudaErrCheck(cudaMemcpy(this->m_dLights, lights.data(), this->m_nLights * sizeof(Light), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMalloc(&this->m_dOriginalLights, this->m_nLights * sizeof(Light)));
    cudaErrCheck(cudaMemcpy(this->m_dOriginalLights, this->m_dLights, this->m_nLights * sizeof(Light),
                            cudaMemcpyDeviceToDevice));

    cudaErrCheck(cudaMalloc(&this->m_dNormals, this->m_nFaces * sizeof(Normals)));
    cudaErrCheck(cudaMemcpy(this->m_dNormals, normals.data(), this->m_nFaces * sizeof(Normals),
                            cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMalloc(&this->m_dOriginalNormals, this->m_nFaces * sizeof(Normals)));
    cudaErrCheck(cudaMemcpy(this->m_dOriginalNormals, this->m_dNormals, this->m_nFaces * sizeof(Normals),
                            cudaMemcpyDeviceToDevice));
}

Renderer::~Renderer()
{
    cudaErrCheck(cudaGraphicsUnregisterResource(this->m_pboRes));
    cudaErrCheck(cudaFree(this->m_dOriginalNormals));
    cudaErrCheck(cudaFree(this->m_dNormals));
    cudaErrCheck(cudaFree(this->m_dOriginalLights));
    cudaErrCheck(cudaFree(this->m_dLights));
    cudaErrCheck(cudaFree(this->m_dOriginalFaces));
    cudaErrCheck(cudaFree(this->m_dFaces));
    cudaErrCheck(cudaFree(this->m_dTexBuf));
}

void Renderer::render()
{
    cudaErrCheck(cudaGraphicsMapResources(1, &this->m_pboRes));
    cudaErrCheck(cudaGraphicsResourceGetMappedPointer(&this->m_dTexBuf, nullptr, this->m_pboRes));
    dim3 gridDim;

    gridDim = dim3((this->m_nFaces + CUDA_BLOCK_DIM_1D.x - 1) / CUDA_BLOCK_DIM_1D.x, 1, 1);
    objectTransformationKernel<<<gridDim, CUDA_BLOCK_DIM_1D>>>(this->m_dOriginalFaces, this->m_dFaces,
                                                               this->m_dOriginalNormals, this->m_dNormals,
                                                               this->m_nFaces,
                                                               this->m_angles, this->m_scale);

    gridDim = dim3((this->m_nLights + CUDA_BLOCK_DIM_1D.x - 1) / CUDA_BLOCK_DIM_1D.x, 1, 1);
    lightRotationKernel<<<gridDim, CUDA_BLOCK_DIM_1D>>>(this->m_dOriginalLights, this->m_dLights,
                                                        this->m_nLights, this->m_lightAngle);
    gridDim = dim3((this->m_width + CUDA_BLOCK_DIM_2D.x - 1) / CUDA_BLOCK_DIM_2D.x,
                   (this->m_height + CUDA_BLOCK_DIM_2D.y - 1) / CUDA_BLOCK_DIM_2D.y, 1);
    shadingKernel<CUDA_BLOCK_DIM_2D.x * CUDA_BLOCK_DIM_2D.y * CUDA_BLOCK_DIM_2D.z><<<gridDim, CUDA_BLOCK_DIM_2D>>>(
        static_cast<uchar3*>(this->m_dTexBuf), this->m_width, this->m_height,
        this->m_dFaces, this->m_dNormals, this->m_nFaces, this->m_dLights, this->m_nLights, this->m_color, this->m_kD,
        this->m_kS, this->m_kA, this->m_alpha, this->camera.pos, this->camera.forwardDir);
    cudaErrCheck(cudaGraphicsUnmapResources(1, &this->m_pboRes));
}

void Renderer::handleKey(int key, float dt)
{
    switch (key)
    {
    case GLFW_KEY_W:
        this->m_angles.y += this->m_rotSpeed * dt;
        break;
    case GLFW_KEY_S:
        this->m_angles.y -= this->m_rotSpeed * dt;
        break;
    case GLFW_KEY_A:
        this->m_angles.x += this->m_rotSpeed * dt;
        break;
    case GLFW_KEY_D:
        this->m_angles.x -= this->m_rotSpeed * dt;
        break;
    case GLFW_KEY_Q:
        this->m_angles.z += this->m_rotSpeed * dt;
        break;
    case GLFW_KEY_E:
        this->m_angles.z -= this->m_rotSpeed * dt;
        break;
    case GLFW_KEY_H:
        this->m_lightAngle += this->m_rotSpeed * dt;
        break;
    case GLFW_KEY_L:
        this->m_lightAngle -= this->m_rotSpeed * dt;
        break;
    case GLFW_KEY_MINUS:
        this->m_scale -= this->m_scaleSpeed * dt;
        break;
    case GLFW_KEY_EQUAL:
        this->m_scale += this->m_scaleSpeed * dt;
        break;
    default:
        this->camera.handleKey(key, dt);
        break;
    }
}