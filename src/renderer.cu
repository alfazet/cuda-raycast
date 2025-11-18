#include "renderer.cuh"

__global__ void shadingKernel(uchar3* texBuf, int width, int height, Triangle* faces, int nFaces, Light* lights,
                              int nLights, Normals* normals, float3 surfaceColor, float kD, float kS, float kA,
                              float alpha, float3 baseCameraPos, float3 rayDir)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= width || ty >= height)
    {
        return;
    }
    float aspect_ratio = width / static_cast<float>(height);
    float x = (2.0f * (tx / static_cast<float>(width)) - 1.0f) * aspect_ratio;
    float y = (2.0f * (ty / static_cast<float>(height)) - 1.0f);
    float3 cameraPos = baseCameraPos + make_float3(x, y, 0.0f);

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

// rotates and scales all faces (together with their normals) and saves them in rotatedFaces/Normals
// TODO: restore the up vector
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

    transFaces[tx].a = vecMatMul3(originalFaces[tx].a, row1x, row2x, row3x);
    transFaces[tx].a = vecMatMul3(transFaces[tx].a, row1y, row2y, row3y);
    transFaces[tx].a = vecMatMul3(transFaces[tx].a, row1z, row2z, row3z);
    transFaces[tx].b = vecMatMul3(originalFaces[tx].b, row1x, row2x, row3x);
    transFaces[tx].b = vecMatMul3(transFaces[tx].b, row1y, row2y, row3y);
    transFaces[tx].b = vecMatMul3(transFaces[tx].b, row1z, row2z, row3z);
    transFaces[tx].c = vecMatMul3(originalFaces[tx].c, row1x, row2x, row3x);
    transFaces[tx].c = vecMatMul3(transFaces[tx].c, row1y, row2y, row3y);
    transFaces[tx].c = vecMatMul3(transFaces[tx].c, row1z, row2z, row3z);

    transFaces[tx].a *= scale;
    transFaces[tx].b *= scale;
    transFaces[tx].c *= scale;

    rotatedNormals[tx].na = vecMatMul3(originalNormals[tx].na, row1x, row2x, row3x);
    rotatedNormals[tx].na = vecMatMul3(rotatedNormals[tx].na, row1y, row2y, row3y);
    rotatedNormals[tx].na = vecMatMul3(rotatedNormals[tx].na, row1z, row2z, row3z);
    rotatedNormals[tx].nb = vecMatMul3(originalNormals[tx].nb, row1x, row2x, row3x);
    rotatedNormals[tx].nb = vecMatMul3(rotatedNormals[tx].nb, row1y, row2y, row3y);
    rotatedNormals[tx].nb = vecMatMul3(rotatedNormals[tx].nb, row1z, row2z, row3z);
    rotatedNormals[tx].nc = vecMatMul3(originalNormals[tx].nc, row1x, row2x, row3x);
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
                                  m_kS{kS}, m_kA{kA}, m_alpha{alpha}, m_color{color}, m_blockDim{CUDA_BLOCK_DIM}
{
    this->m_gridDim = dim3((this->m_width + this->m_blockDim.x - 1) / this->m_blockDim.x,
                           (this->m_height + this->m_blockDim.y - 1) / this->m_blockDim.y, 1);

    // TODO: make faces, normals and lights SoA

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
    objectTransformationKernel<<<this->m_gridDim, this->m_blockDim>>>(this->m_dOriginalFaces, this->m_dFaces,
                                                                      this->m_dOriginalNormals, this->m_dNormals,
                                                                      this->m_nFaces,
                                                                      this->m_angles, this->m_scale);
    lightRotationKernel<<<this->m_gridDim, this->m_blockDim>>>(this->m_dOriginalLights, this->m_dLights,
                                                               this->m_nLights, this->m_lightAngle);
    // TODO: sort faces by (x, y) so that we can consider only relevant triangles in a block by moving them to shmem
    shadingKernel<<<this->m_gridDim, this->m_blockDim>>>(static_cast<uchar3*>(this->m_dTexBuf), this->m_width,
                                                         this->m_height, this->m_dFaces, this->m_nFaces,
                                                         this->m_dLights, this->m_nLights, this->m_dNormals,
                                                         this->m_color, this->m_kD,
                                                         this->m_kS, this->m_kA, this->m_alpha, this->camera.pos,
                                                         this->camera.forwardDir);
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