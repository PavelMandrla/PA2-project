#include "Particles.cuh"

#include <random>
#include <chrono>
#include <cudaDefs.h>

constexpr unsigned int TPB_1D = 1024;
constexpr unsigned int TPB_8D = 8;

__constant__ __device__ float cSpeedFactor;
__constant__ __device__ float cRadiusSQ;
__constant__ __device__ int2 cTexDim;
__constant__ __device__ int2 cOverlayDim;


Particles::Particles(shared_ptr<Settings> settings, shared_ptr<HeightMap> hMap) {
    this->settings = settings;
    this->hMap = hMap;

    this->activeLeaders = settings->leaders;
    this->activeFollowers = settings->followers;

    this->generateParticles();
    // COPY CONSTS TO CONST MAMEORY
    float radiusSq = pow(settings->leaderRadius, 2);
    int2 texDim {int(hMap->glData.imageWidth), int(hMap->glData.imageHeight)};
    int2 overlayDim {settings->heightmapGridX, settings->heightmapGridY};
    checkCudaErrors(cudaMemcpyToSymbol((const void*)&cRadiusSQ,     &radiusSq,              sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol((const void*)&cSpeedFactor,  &settings->speedFactor, sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol((const void*)&cTexDim,       &texDim,                sizeof(int2)));
    checkCudaErrors(cudaMemcpyToSymbol((const void*)&cOverlayDim,   &overlayDim,            sizeof(int2)));


    this->status = cublasStatus_t();
    this->handle = cublasHandle_t();
    this->status = cublasCreate(&handle) ;

#ifdef USE_CUBLAS
    unsigned int onesCount = 2 * (settings->leaders > settings->followers ? settings->leaders : settings->followers);
    cudaMalloc((void**)&this->dOnes, sizeof(float) * onesCount);
    cudaMemcpy(dOnes, vector<float>(onesCount, 1.0f).data(), sizeof(float) * onesCount, cudaMemcpyHostToDevice);
#endif
    checkCudaErrors(cudaMalloc((void**)&this->dDistances, activeLeaders * activeFollowers * sizeof(float)));
}

Particles::~Particles() {
    status = cublasDestroy(handle);
    if (this->dLeaderPos) cudaFree(this->dLeaderPos);
    if (this->dLeaderDir) cudaFree(this->dLeaderDir);
    if (this->dFollowerPos) cudaFree(this->dFollowerPos);
    if (this->dFollowerPosNext) cudaFree(this->dFollowerPosNext);
    if (this->dActiveFollowersNext) cudaFree(this->dActiveFollowersNext);
    if (this->dDistances) cudaFree(this->dDistances);

#ifdef USE_CUBLAS
    if (this->dLeadersPosSq) cudaFree(this->dLeadersPosSq);
    if (this->dFollowersPosSq) cudaFree(this->dFollowersPosSq);
    if (this->dOnes) cudaFree(this->dOnes);
#endif
}

vector<float2> Particles::generatePositions(int n) {
    vector<float2> pos;
    std::mt19937 generator(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    std::uniform_real_distribution<float> dis(0.001f, 0.999f);

    for (int i = 0; i < n; i++) {
        // POSITION ON MAP
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
    checkCudaErrors(cudaMalloc((void**)&dFollowerPosNext, settings->followers * sizeof(float2)));
    checkCudaErrors(cudaMemcpy(dFollowerPos, generatePositions(settings->followers).data(), settings->followers * sizeof(float2), cudaMemcpyHostToDevice));
#ifdef USE_CUBLAS
    //positions SQ
    checkCudaErrors(cudaMalloc((void**)&dLeadersPosSq, 2 * settings->leaders * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&dFollowersPosSq, 2 * settings->followers * sizeof(float)));
#endif
    // FOLLOWERS - alive counter
    checkCudaErrors(cudaMalloc((void**)&dActiveFollowersNext, sizeof(unsigned int)));

}

#ifdef USE_CUBLAS

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
    dim3 block(128,1,1);
    dim3 grid((activeParticles + TPB_1D - 1) / TPB_1D);
    createSquareMatrix<<<block, grid>>>(dParticles, activeParticles, dPositionsSq);
}

void Particles::calculateDistances() {
    // SQUARE OF POSITIONS
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

#else
__global__ void calculateDistanceMatrix(float2* dLeaders, unsigned int leaderCount, float2* dFollowers, int followerCount, float* dstMat) {
    __shared__ float2 sLeaders[6000];
    if (threadIdx.x < leaderCount) {
        sLeaders[threadIdx.x] = dLeaders[threadIdx.x];
    }
    __syncthreads();

    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int jump = gridDim.x * blockDim.x;

    float2* follower = &dFollowers[idx];
    while (idx < followerCount) {
        float *dst = dstMat + idx * leaderCount;
        float2 f = *follower;

        for (int i = 0; i < leaderCount; i++) {
            float2 l = sLeaders[i];
            *dst = pow(l.x - f.x, 2) + pow(l.y - f.y, 2);

            dst++;
        }

        follower += jump;
        idx += jump;
    }
}

void Particles::calculateDistances() {
    dim3 block(TPB_1D, 1, 1);
    dim3 grid((activeFollowers + TPB_1D - 1) / TPB_1D, 1, 1);
    calculateDistanceMatrix<<<grid, block>>>(dLeaderPos, activeLeaders, dFollowerPos, activeFollowers, dDistances);
}

#endif

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

__device__ __forceinline__ void renderParticle(int x, int y, uchar3 color, unsigned char* pbo) {
    int nX = cOverlayDim.x * x / cTexDim.x;
    int nY = cOverlayDim.y * y / cTexDim.y;

    unsigned int pboIdx = ((nY * cOverlayDim.x) + nX) * 4;
    pbo[pboIdx++] = color.x;
    pbo[pboIdx++] = color.y;
    pbo[pboIdx++] = color.z;
    pbo[pboIdx]   = 255;
}

__device__ __forceinline__ float sigmoid(float x, const float c1 = 6.0f, const float c2 = 1.0f, const float max = 2.0f) {
    return max / (1 + exp(-c1 * (x - c2)));
}

template<bool normalizeVector>__device__ __forceinline__ float2 getNewParticlePos(float2 pos, float2 dir, float dirLen, const cudaTextureObject_t srcTex) {
    if (normalizeVector) {
        dir.x /= dirLen;
        dir.y /= dirLen;
    }
    float2 nPos = float2 {pos.x + dir.x * cSpeedFactor,
                          pos.y + dir.y * cSpeedFactor };

    float dH = 1.0f + (float(tex2D<float>(srcTex, pos.x, pos.y)) - float(tex2D<float>(srcTex, nPos.x, nPos.y)));
    //float dH = 1.0f + (float(tex2D<uchar1>(srcTex, pos.x, pos.y).x) - float(tex2D<uchar1>(srcTex, nPos.x, nPos.y).x)) / 256.0f;
    return float2 {pos.x + dir.x * cSpeedFactor * sigmoid(dH), pos.y + dir.y * cSpeedFactor * sigmoid(dH)};
    //return float2 {pos.x + dir.x * cSpeedFactor * dH, pos.y + dir.y * cSpeedFactor * dH};

}


__global__ void moveParticles_Leaders(const cudaTextureObject_t srcTex,
                                      float2* particlePos,
                                      float2* particleDir,
                                      unsigned int particleCount,
                                      unsigned char* pbo) {
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int jump = blockDim.x * gridDim.x;

    float2* pos = &particlePos[tx];
    float2* dir = &particleDir[tx];
    while (tx < particleCount) {
        float2 nPos = getNewParticlePos<false>(*pos, *dir, 1, srcTex);
        if (nPos.x < 0 || nPos.x >= cTexDim.x) {
            dir->x *= -1;
            nPos = getNewParticlePos<false>(*pos, *dir, 1, srcTex);
        }
        if (nPos.y < 0 || nPos.y >= cTexDim.y) {
            dir->y *= -1;
            nPos = getNewParticlePos<false>(*pos, *dir, 1, srcTex);
        }
        *pos = nPos;

        renderParticle(floor(nPos.x), floor(nPos.y), uchar3 {255, 0, 0}, pbo);

        pos += jump;
        dir += jump;
        tx += jump;
    }
}

void Particles::moveLeaders(unsigned char* pboData) {
    dim3 block(TPB_1D, 1, 1);
    dim3 grid((activeLeaders + TPB_1D - 1) / TPB_1D, 1, 1);
    moveParticles_Leaders<<<grid, block>>>(hMap->cudaData.texObj, dLeaderPos, dLeaderDir, activeLeaders, pboData);
}


__global__ void moveParticles_Followers(const cudaTextureObject_t srcTex,
                                        float2* particlePos,
                                        float2* particlePos_next,
                                        unsigned int particleCount,
                                        float2* leaderPos,
                                        unsigned int leaderCount,
                                        float* dDistances,
                                        unsigned int* dTerminatedCounter,
                                        unsigned char* pbo) {
    __shared__ float2 sLeaderPos[6000];

    if (threadIdx.x < leaderCount) {
        sLeaderPos[threadIdx.x] = leaderPos[threadIdx.x];
    }
    __syncthreads();

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
        float2 clLeaderPos = sLeaderPos[clLeaderI];
        float2 nPos;
        // MOVE TOWARDS LEADER
        if (clLeaderDst > 0) { // PREVENT DIVISION BY 0
            float2 dir = {clLeaderPos.x - pos->x, clLeaderPos.y - pos->y};
            nPos = getNewParticlePos<true>(*pos, dir, sqrt(clLeaderDst), srcTex);
        } else if (clLeaderDst < 0) {
            nPos = float2 {10,10};
        } else {
            nPos = *pos;
        }

        //CALCULATE NEW DISTANCE TO LEADER
        float newDist = pow(clLeaderPos.x - nPos.x, 2) + pow(clLeaderPos.y - nPos.y, 2);
        if (newDist > cRadiusSQ) {
            unsigned int iPosNext = atomicAdd(dTerminatedCounter, 1);
            particlePos_next[iPosNext] = nPos;
        }

        renderParticle(floor(pos->x), floor(pos->y), uchar3 {0, 0, 255}, pbo);

        pos += jump;
        idx += jump;
    }
}

void Particles::moveFollowers(unsigned char* pboData) {
    //RESET TERMINATED COUNTER
    checkCudaErrors(cudaMemcpy(dActiveFollowersNext, vector<unsigned int>(1, 0).data(), sizeof(unsigned int), cudaMemcpyHostToDevice));

    dim3 block(TPB_1D, 1, 1);
    dim3 grid((activeFollowers + TPB_1D - 1) / TPB_1D, 1, 1);

    moveParticles_Followers<<<grid, block>>>(hMap->cudaData.texObj, dFollowerPos, dFollowerPosNext, activeFollowers, dLeaderPos, activeLeaders, dDistances, dActiveFollowersNext, pboData);

    unsigned int hActiveFollowersNext;
    checkCudaErrors(cudaMemcpy(&hActiveFollowersNext, dActiveFollowersNext, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("%d\n", hActiveFollowersNext);
    this->activeFollowers = hActiveFollowersNext;

    std::swap(dFollowerPos, dFollowerPosNext);
}

void Particles::move() {
    // REGISTER HEIGHTMAP RESOURCE
    checkCudaErrors(cudaGraphicsMapResources(1, &hMap->cudaData.texResource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&hMap->cudaData.texArrayData, hMap->cudaData.texResource, 0, 0));

    // REGISTER OVERLAY RESOURCE
    checkCudaErrors(cudaGraphicsMapResources(1, &hMap->cudaData.pboResource, 0));
    unsigned char* pboData;
    size_t pboSize;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&pboData, &pboSize, hMap->cudaData.pboResource));

    {   // CLEAR PBO
        dim3 block(TPB_8D, TPB_8D, 1);
        dim3 grid((settings->heightmapGridX+ TPB_8D - 1) / TPB_8D, (settings->heightmapGridY + TPB_8D - 1) / TPB_8D, 1);
        clearPBO<<<grid, block>>>(pboData, settings->heightmapGridX, settings->heightmapGridY);
    };

    this->moveLeaders(pboData);
    this->calculateDistances();
    this->moveFollowers(pboData);

    // UNREGISTER OVERLAY RESOURCE
    checkCudaErrors(cudaGraphicsUnmapResources(1, &hMap->cudaData.pboResource, 0));
    // UNREGISTER HEIGHTMAP RESOURCE
    checkCudaErrors(cudaGraphicsUnmapResources(1, &hMap->cudaData.texResource, 0));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, hMap->overlayPboID);
    auto err = glGetError();
    glBindTexture(GL_TEXTURE_2D, hMap->overlayTexId);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, settings->heightmapGridX, settings->heightmapGridY, GL_RGBA, GL_UNSIGNED_BYTE, NULL);   //Source parameter is NULL, Data is coming from a PBO, not host memory
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}









