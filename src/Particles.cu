#include "Particles.cuh"

#include <random>
#include <chrono>
#include <cudaDefs.h>

Particles::Particles(shared_ptr<Settings> settings, shared_ptr<HeightMap> hMap) {
    this->settings = settings;
    this->hMap = hMap;

    this->activeLeaders = settings->leaders;
    this->activeFollowers = settings->followers;

    this->generateParticles();

    this->status = cublasStatus_t();
    this->handle = cublasHandle_t();
    this->status = cublasCreate(&handle) ;

    unsigned int onesCount = 2 * (settings->leaders > settings->followers ? settings->leaders : settings->followers);
    cudaMalloc((void**)&this->dOnes, sizeof(float) * onesCount);
    cudaMemcpy(dOnes, vector<float>(onesCount, 1.0f).data(), sizeof(float) * onesCount, cudaMemcpyHostToDevice);

    checkCudaErrors(cudaMalloc((void**)&this->dDistances, activeLeaders * activeFollowers * sizeof(float)));
}

Particles::~Particles() {
    status = cublasDestroy(handle);
    if (this->dLeaderPos) cudaFree(this->dLeaderPos);
    if (this->dLeaderDir) cudaFree(this->dLeaderDir);
    if (this->dFollowerPos) cudaFree(this->dFollowerPos);
    if (this->dDistances) cudaFree(this->dDistances);
}

vector<float2> Particles::generatePositions(int n) {
    vector<float2> pos;
    std::mt19937 generator(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    std::uniform_real_distribution<float> dis(0.001f, 0.999f);

    //float dW = imgWidth / float(n);
    //float dH = imgHeight / float(n);
    for (int i = 0; i < n; i++) {
        // POSITION ON MAP
        //pos.push_back(float2 {i * dW, i * dH});
        pos.push_back(float2 {dis(generator) * float(hMap->glData.imageWidth), dis(generator) * float(hMap->glData.imageHeight)});
    }
    return pos;
}

vector<float2> Particles::generateDirections(int n) {
    vector<float2> dirs;
    std::mt19937 generator(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    std::uniform_real_distribution<float> disVel(-1, 1);

    for (int i = 0; i < n; i++) {
        // NORMALIZED MOVEMENT DIRECTION
        float2 dir { disVel(generator), disVel(generator) };
        float dirL = sqrt(pow(dir.x, 2) + pow(dir.y, 2));
        dir.x /= dirL;
        dir.y /= dirL;
        dirs.push_back(dir);
    }
    return dirs;
}

void Particles::generateParticles() {
    // LEADERS - positions
    checkCudaErrors(cudaMalloc((void**)&dLeaderPos, settings->leaders * sizeof(float2)));
    checkCudaErrors(cudaMemcpy(dLeaderPos, generatePositions(settings->leaders).data(), settings->leaders * sizeof(float2), cudaMemcpyHostToDevice));
    // LEADERS - directions
    checkCudaErrors(cudaMalloc((void**)&dLeaderDir, settings->leaders * sizeof(float2)));
    checkCudaErrors(cudaMemcpy(dLeaderDir, generateDirections(settings->leaders).data(), settings->leaders * sizeof(float2), cudaMemcpyHostToDevice));
    // FOLLOWERS - positions
    checkCudaErrors(cudaMalloc((void**)&dFollowerPos, settings->followers * sizeof(float2)));
    checkCudaErrors(cudaMemcpy(dFollowerPos, generatePositions(settings->followers).data(), settings->followers * sizeof(float2), cudaMemcpyHostToDevice));

}


__global__ void clearPBO(unsigned char* pbo, const unsigned int pboWidth, const unsigned int pboHeight) {
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= pboWidth || ty > pboHeight) return;
    unsigned int pboIdx = ((ty * pboWidth) + tx) * 4;

    pbo[pboIdx++] = 0;
    pbo[pboIdx++] = 0;
    pbo[pboIdx++] = 0;
    pbo[pboIdx]   = 0;
}

__device__ __forceinline__ void renderPixel(int x, int y, uchar3 color, const unsigned int width, const unsigned int height, unsigned char* pbo) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        unsigned int pboIdx = ((y * width) + x) * 4;
        pbo[pboIdx++] = color.x;
        pbo[pboIdx++] = color.y;
        pbo[pboIdx++] = color.z;
        pbo[pboIdx]   = 255;
    }
}

__global__ void renderParticles(uchar3 color, float2* particles, int particleCount, unsigned char* pbo, const unsigned int pboWidth, const unsigned int pboHeight) {
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int jump = blockDim.x * gridDim.x;

    while (tx < particleCount) {
        float2 p = particles[tx];
        int x = floor(p.x);
        int y = floor(p.y); //TODO -> is origing OK?

        for (int dX = -10; dX <= 10; dX++) {
            for (int dY = -10; dY <= 10; dY++) {
                renderPixel(x+dX, y + dY, color, pboWidth, pboHeight, pbo);
            }
        }
        tx += jump;
    }
}

void Particles::renderToOverlay() {
    checkCudaErrors(cudaGraphicsMapResources(1, &hMap->cudaData.pboResource, 0));
    unsigned char* pboData;
    size_t pboSize;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&pboData, &pboSize, hMap->cudaData.pboResource));

    {   // CLEAR PBO
        constexpr unsigned int TPB_1D = 8; //TODO -> define somewhere TPB_1D
        dim3 block(TPB_1D, TPB_1D, 1);
        dim3 grid((hMap->glData.imageWidth + TPB_1D - 1) / TPB_1D, (hMap->glData.imageHeight + TPB_1D - 1) / TPB_1D, 1);
        clearPBO<<<grid, block>>>(pboData, hMap->glData.imageWidth, hMap->glData.imageHeight);
    };

    {   // PUT PARTCLES INTO PBO
        //TODO -> adjust block and grid sizes
        constexpr unsigned int TPB_1D = 128; //TODO -> define somewhere TPB_1D
        dim3 block(128, 1, 1);
        dim3 grid((hMap->glData.imageWidth + TPB_1D - 1) / TPB_1D, 1, 1);
        renderParticles<<<grid, block>>>(Particles::followerColor,  dFollowerPos, settings->followers, pboData, hMap->glData.imageWidth, hMap->glData.imageHeight);
        renderParticles<<<grid, block>>>(Particles::leaderColor,    dLeaderPos,   settings->leaders,   pboData, hMap->glData.imageWidth, hMap->glData.imageHeight);
    };

    checkCudaErrors(cudaGraphicsUnmapResources(1, &hMap->cudaData.pboResource, 0));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, hMap->glData.pboID);
    auto err = glGetError();
    glBindTexture(GL_TEXTURE_2D, hMap->overlayTexId);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, hMap->glData.imageWidth, hMap->glData.imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);   //Source parameter is NULL, Data is coming from a PBO, not host memory
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}


__global__ void createSquareMatrix(float2* particles, int particleCount, float* dst) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int jump = gridDim.x * blockDim.x;

    float2 * particle = &particles[idx];
    float* row = &dst[2*idx];
    while (idx < particleCount) {
        row[0] = particle->x * particle->x;
        row[1] = particle->y * particle->y;

        particle += jump;
        row += jump;
        idx += jump;
    }
}

inline void squarePositions(float* &dPositionsSq, float2* dParticles, unsigned int activeParticles) {
    checkCudaErrors(cudaMalloc((void**)&dPositionsSq, 2 * activeParticles * sizeof(float)));
    constexpr unsigned int TPB_1D = 128;
    dim3 block(128,1,1); //TODO -> change TPB1D?
    dim3 grid((activeParticles + TPB_1D - 1) / TPB_1D);
    createSquareMatrix<<<block, grid>>>(dParticles, activeParticles, dPositionsSq);
}

void Particles::calculateDistances() {
    // SQUARE OF POSITIONS
    float* dLeadersPosSq;
    float* dFollowersPosSq;
    squarePositions(dLeadersPosSq, dLeaderPos, activeLeaders);
    squarePositions(dFollowersPosSq, dFollowerPos, activeFollowers);

    float alpha = 1.0f;
    float beta = 0.0f;
    auto response = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                activeLeaders, activeFollowers, 2,  //  M, N, K
                                &alpha,
                                dOnes, 2,
                                dFollowersPosSq, 2,
                                &beta,
                                dDistances, activeLeaders);

    //alpha = 1.0f;
    beta = 1.0f;
    response = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           activeLeaders, activeFollowers, 2,  //  M, N, K
                           &alpha,
                           dLeadersPosSq, 2,
                           dOnes, 2,
                           &beta,
                           dDistances, activeLeaders);

    alpha = -2.0f;
    //beta = 1.0f;
    response = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                activeLeaders, activeFollowers, 2,  //  M, N, K
                                &alpha,
                                (float*)dLeaderPos, 2,
                                (float*)dFollowerPos, 2,
                                &beta,
                                dDistances, activeLeaders);

    //checkDeviceMatrix<float>(dDistances,sizeof(float) * activeLeaders, activeFollowers, activeLeaders, "%f ", "M");

}

__global__ void moveParticles_Followers(float2* particlePos, unsigned int particleCount, float2* leaderPos, unsigned int leaderCount, float* dDistances) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int jump = gridDim.x * blockDim.x;

    float2 *pos = &particlePos[idx];

    while (idx < particleCount) {
        //FIND CLOSEST LEADER
        float *dst = dDistances + idx * leaderCount;
        int clLeaderI = 0;
        float clLeaderDst = *dst;
        for (int i = 1; i < leaderCount; i++) {
            if (dst[i] < clLeaderDst) {
                clLeaderDst = dst[i];
                clLeaderI = i;
            }
        }

        if (clLeaderDst > 0) {
            // MOVE TO LEADER
            float2 clLeaderPos = leaderPos[clLeaderI];
            //float2 clLeaderPos = float2 {400, 400};
            float2 dir = {(clLeaderPos.x - pos->x) / sqrt(clLeaderDst), (clLeaderPos.y - pos->y) / sqrt(clLeaderDst)};

            *pos = float2{pos->x + dir.x, pos->y + dir.y};
        }


        pos += jump;
        idx += jump;
    }
}

void Particles::moveFollowers() {
    constexpr unsigned int TPB_1D = 128; //TODO -> define somewhere TPB_1D
    dim3 block(TPB_1D, 1, 1);
    dim3 grid((activeLeaders + TPB_1D - 1) / TPB_1D, 1, 1);
    moveParticles_Followers<<<grid, block>>>(dFollowerPos, activeFollowers, dLeaderPos, activeLeaders, dDistances);
}

__global__ void moveParticles_Leaders(const cudaTextureObject_t srcTex, float2* particlePos, float2* particleDir, unsigned int particleCount, int imgW, int imgH) {
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int jump = blockDim.x * gridDim.x;

    float2* pos = &particlePos[tx];
    float2* dir = &particleDir[tx];
    while (tx < particleCount) {
        float2 nPos = float2 { pos->x + dir->x, pos->y + dir->y };
        if (nPos.x < 0 || nPos.x >= imgW) {
            dir->x *= -1;
            nPos.x = pos->x + dir->x;
        }
        if (nPos.y < 0 || nPos.y >= imgH) {
            dir->y *= -1;
            nPos.y = pos->y + dir->y;
        }

        float dH = 1.0f + (float(tex2D<uchar1>(srcTex, nPos.x, nPos.y).x) - float(tex2D<uchar1>(srcTex, pos->x, pos->y).x)) / 256.0f;
        float speedFactor = 2 * exp(dH - 2.0f);
        //*pos = nPos;

        if (speedFactor > 0.6) {
            *pos = float2 { pos->x + speedFactor*dir->x, pos->y + speedFactor*dir->y };
        }


        pos += jump;
        dir += jump;
        tx += jump;
    }
}

void Particles::moveLeaders() {
    constexpr unsigned int TPB_1D = 128; //TODO -> define somewhere TPB_1D
    dim3 block(TPB_1D, 1, 1);
    dim3 grid((activeLeaders + TPB_1D - 1) / TPB_1D, 1, 1);
    moveParticles_Leaders<<<grid, block>>>(hMap->cudaData.texObj, dLeaderPos, dLeaderDir, activeLeaders, hMap->glData.imageWidth, hMap->glData.imageHeight);
}

void Particles::move() {
    checkCudaErrors(cudaGraphicsMapResources(1, &hMap->cudaData.texResource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&hMap->cudaData.texArrayData, hMap->cudaData.texResource, 0, 0));

    this->moveLeaders();
    this->calculateDistances();
    this->moveFollowers();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &hMap->cudaData.texResource, 0));
}







