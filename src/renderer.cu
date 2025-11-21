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

template <int N> __global__ void shadingKernel(uchar3* texBuf, int width, int height, TriangleSOA faces,
                                               NormalsSOA normals,
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
        reachableIdxs[tidxInBlock] = -1;
        if (faceIdx < nFaces)
        {
            float3 vA = faces.a[faceIdx], vB = faces.b[faceIdx], vC = faces.c[faceIdx];
            float minX = min(vA.x, min(vB.x, vC.x));
            float minY = min(vA.y, min(vB.y, vC.y));
            float maxX = max(vA.x, max(vB.x, vC.x));
            float maxY = max(vA.y, max(vB.y, vC.y));
            isReachable = minX <= blockMaxX && minY <= blockMaxY && maxX >= blockMinX && maxY >= blockMinY;
            if (isReachable)
            {
                reachableFaces[tidxInBlock] = Triangle{.a = vA, .b = vB, .c = vC};
                reachableIdxs[tidxInBlock] = faceIdx;
            }
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
            hitPointBary.x * normals.na[hitIdx] + hitPointBary.y * normals.nb[hitIdx] + hitPointBary.z * normals.nc[
                hitIdx]);
        float3 hitPoint = cameraPos + minT * rayDir;
        texBuf[ty * width + tx] = computeColor(hitPoint, normal, cameraPos, lights, nLights, surfaceColor, kD, kS, kA,
                                               alpha);
    }
}

// TODO: quaternions
// rotates and scales all faces (together with their normals)
__global__ void objectTransformationKernel(TriangleSOA originalFaces, TriangleSOA transFaces,
                                           NormalsSOA originalNormals,
                                           NormalsSOA rotatedNormals, int nFaces, float3 angles, float scale)
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

    transFaces.a[tx] = originalFaces.a[tx];
    transFaces.b[tx] = originalFaces.b[tx];
    transFaces.c[tx] = originalFaces.c[tx];
    rotatedNormals.na[tx] = originalNormals.na[tx];
    rotatedNormals.nb[tx] = originalNormals.nb[tx];
    rotatedNormals.nc[tx] = originalNormals.nc[tx];

    transFaces.a[tx] = vecMatMul3(transFaces.a[tx], row1x, row2x, row3x);
    transFaces.a[tx] = vecMatMul3(transFaces.a[tx], row1y, row2y, row3y);
    transFaces.a[tx] = vecMatMul3(transFaces.a[tx], row1z, row2z, row3z);
    transFaces.b[tx] = vecMatMul3(transFaces.b[tx], row1x, row2x, row3x);
    transFaces.b[tx] = vecMatMul3(transFaces.b[tx], row1y, row2y, row3y);
    transFaces.b[tx] = vecMatMul3(transFaces.b[tx], row1z, row2z, row3z);
    transFaces.c[tx] = vecMatMul3(transFaces.c[tx], row1x, row2x, row3x);
    transFaces.c[tx] = vecMatMul3(transFaces.c[tx], row1y, row2y, row3y);
    transFaces.c[tx] = vecMatMul3(transFaces.c[tx], row1z, row2z, row3z);

    transFaces.a[tx] *= scale;
    transFaces.b[tx] *= scale;
    transFaces.c[tx] *= scale;

    rotatedNormals.na[tx] = vecMatMul3(rotatedNormals.na[tx], row1x, row2x, row3x);
    rotatedNormals.na[tx] = vecMatMul3(rotatedNormals.na[tx], row1y, row2y, row3y);
    rotatedNormals.na[tx] = vecMatMul3(rotatedNormals.na[tx], row1z, row2z, row3z);
    rotatedNormals.nb[tx] = vecMatMul3(rotatedNormals.nb[tx], row1x, row2x, row3x);
    rotatedNormals.nb[tx] = vecMatMul3(rotatedNormals.nb[tx], row1y, row2y, row3y);
    rotatedNormals.nb[tx] = vecMatMul3(rotatedNormals.nb[tx], row1z, row2z, row3z);
    rotatedNormals.nc[tx] = vecMatMul3(rotatedNormals.nc[tx], row1x, row2x, row3x);
    rotatedNormals.nc[tx] = vecMatMul3(rotatedNormals.nc[tx], row1y, row2y, row3y);
    rotatedNormals.nc[tx] = vecMatMul3(rotatedNormals.nc[tx], row1z, row2z, row3z);
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

void destructureFaces(std::vector<Triangle>& faces, std::vector<float3>& a, std::vector<float3>& b,
                      std::vector<float3>& c)
{
    int n = faces.size();
    a.resize(n);
    b.resize(n);
    c.resize(n);
    for (int i = 0; i < n; i++)
    {
        a[i] = faces[i].a;
        b[i] = faces[i].b;
        c[i] = faces[i].c;
    }
}

void destructureNormals(std::vector<Normals>& normals, std::vector<float3>& a, std::vector<float3>& b,
                        std::vector<float3>& c)
{
    int n = normals.size();
    a.resize(n);
    b.resize(n);
    c.resize(n);
    for (int i = 0; i < n; i++)
    {
        a[i] = normals[i].na;
        b[i] = normals[i].nb;
        c[i] = normals[i].nc;
    }
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

    std::vector<float3> facesA, facesB, facesC;
    destructureFaces(faces, facesA, facesB, facesC);
    cudaErrCheck(cudaMalloc(&this->m_dFaces.a, this->m_nFaces * sizeof(float3)));
    cudaErrCheck(cudaMalloc(&this->m_dFaces.b, this->m_nFaces * sizeof(float3)));
    cudaErrCheck(cudaMalloc(&this->m_dFaces.c, this->m_nFaces * sizeof(float3)));
    cudaErrCheck(cudaMemcpy(this->m_dFaces.a, facesA.data(), this->m_nFaces * sizeof(float3),
                            cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(this->m_dFaces.b, facesB.data(), this->m_nFaces * sizeof(float3),
                            cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(this->m_dFaces.c, facesC.data(), this->m_nFaces * sizeof(float3),
                            cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMalloc(&this->m_dOriginalFaces.a, this->m_nFaces * sizeof(float3)));
    cudaErrCheck(cudaMalloc(&this->m_dOriginalFaces.b, this->m_nFaces * sizeof(float3)));
    cudaErrCheck(cudaMalloc(&this->m_dOriginalFaces.c, this->m_nFaces * sizeof(float3)));
    cudaErrCheck(cudaMemcpy(this->m_dOriginalFaces.a, this->m_dFaces.a, this->m_nFaces * sizeof(float3),
                            cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(this->m_dOriginalFaces.b, this->m_dFaces.b, this->m_nFaces * sizeof(float3),
                            cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(this->m_dOriginalFaces.c, this->m_dFaces.c, this->m_nFaces * sizeof(float3),
                            cudaMemcpyDeviceToDevice));

    cudaErrCheck(cudaMalloc(&this->m_dLights, this->m_nLights * sizeof(Light)));
    cudaErrCheck(cudaMemcpy(this->m_dLights, lights.data(), this->m_nLights * sizeof(Light), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMalloc(&this->m_dOriginalLights, this->m_nLights * sizeof(Light)));
    cudaErrCheck(cudaMemcpy(this->m_dOriginalLights, this->m_dLights, this->m_nLights * sizeof(Light),
                            cudaMemcpyDeviceToDevice));

    std::vector<float3> normalsA, normalsB, normalsC;
    destructureNormals(normals, normalsA, normalsB, normalsC);
    cudaErrCheck(cudaMalloc(&this->m_dNormals.na, this->m_nFaces * sizeof(float3)));
    cudaErrCheck(cudaMalloc(&this->m_dNormals.nb, this->m_nFaces * sizeof(float3)));
    cudaErrCheck(cudaMalloc(&this->m_dNormals.nc, this->m_nFaces * sizeof(float3)));
    cudaErrCheck(cudaMemcpy(this->m_dNormals.na, normalsA.data(), this->m_nFaces * sizeof(float3),
                            cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(this->m_dNormals.nb, normalsB.data(), this->m_nFaces * sizeof(float3),
                            cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(this->m_dNormals.nc, normalsC.data(), this->m_nFaces * sizeof(float3),
                            cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMalloc(&this->m_dOriginalNormals.na, this->m_nFaces * sizeof(float3)));
    cudaErrCheck(cudaMalloc(&this->m_dOriginalNormals.nb, this->m_nFaces * sizeof(float3)));
    cudaErrCheck(cudaMalloc(&this->m_dOriginalNormals.nc, this->m_nFaces * sizeof(float3)));
    cudaErrCheck(cudaMemcpy(this->m_dOriginalNormals.na, this->m_dNormals.na, this->m_nFaces * sizeof(float3),
                            cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(this->m_dOriginalNormals.nb, this->m_dNormals.nb, this->m_nFaces * sizeof(float3),
                            cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(this->m_dOriginalNormals.nc, this->m_dNormals.nc, this->m_nFaces * sizeof(float3),
                            cudaMemcpyDeviceToDevice));
}

Renderer::~Renderer()
{
    cudaErrCheck(cudaGraphicsUnregisterResource(this->m_pboRes));
    // cudaErrCheck(cudaFree(this->m_dOriginalNormals));
    // cudaErrCheck(cudaFree(this->m_dNormals));
    cudaErrCheck(cudaFree(this->m_dOriginalLights));
    cudaErrCheck(cudaFree(this->m_dLights));
    cudaErrCheck(cudaFree(this->m_dOriginalFaces.a));
    cudaErrCheck(cudaFree(this->m_dOriginalFaces.b));
    cudaErrCheck(cudaFree(this->m_dOriginalFaces.c));
    cudaErrCheck(cudaFree(this->m_dFaces.a));
    cudaErrCheck(cudaFree(this->m_dFaces.b));
    cudaErrCheck(cudaFree(this->m_dFaces.c));
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
        this->m_scale = max(0.001f, this->m_scale - this->m_scaleSpeed * dt);
        break;
    case GLFW_KEY_EQUAL:
        this->m_scale += this->m_scaleSpeed * dt;
        break;
    default:
        this->camera.handleKey(key, dt);
        break;
    }
}
