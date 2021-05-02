#include "Particles.cuh"

#include <random>
#include <chrono>
#include <cudaDefs.h>

Particles::Particles(shared_ptr<Settings> settings, shared_ptr<HeightMap> hMap) {
    this->settings = settings;
    this->hMap = hMap;

    this->activeLeaders = settings->leaders;
    this->activeFollowers = settings->followers;

    float w = float(hMap->glData.imageWidth);
    float h = float(hMap->glData.imageHeight);
    this->generateOnGPU(settings->leaders, w, h, this->dLeaderPos, this->dLeaderVel);
    this->generateOnGPU(settings->followers, w, h, this->dFollowerPos, this->dFollowerVel);

    this->status = cublasStatus_t();
    this->handle = cublasHandle_t();
    this->status = cublasCreate(&handle) ;

    unsigned int onesCount = 2 * (settings->leaders > settings->followers ? settings->leaders : settings->followers);
    cudaMalloc((void**)&this->dOnes, sizeof(float) * onesCount);
    cudaMemcpy(dOnes, vector<float>(onesCount, 1.0f).data(), sizeof(float) * onesCount, cudaMemcpyHostToDevice);

    checkCudaErrors(cudaMalloc((void**)&this->dDistances, activeLeaders * activeFollowers * sizeof(float)));
    //cudaMalloc( (void**)&dDistances, settings->leaders * settings->followers * sizeof(float));
}

Particles::~Particles() {
    status = cublasDestroy(handle);
    if (this->dLeaderPos) cudaFree(this->dLeaderPos);
    if (this->dLeaderVel) cudaFree(this->dLeaderVel);
    if (this->dFollowerPos) cudaFree(this->dFollowerPos);
    if (this->dFollowerVel) cudaFree(this->dFollowerVel);
    if (this->dDistances) cudaFree(this->dDistances);
}

pair<vector<float2>, vector<float2>> Particles::generate(int n, float imgWidth, float imgHeight) {
    vector<float2> pos;
    vector<float2> vel;

    std::mt19937 generator(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    std::uniform_real_distribution<float> disVel(-1, 1);

    float dW = imgWidth / float(n);
    float dH = imgHeight / float(n);
    for (int i = 0; i < n; i++) {
        //pos.push_back(float2 {i * dW, i * dH});
        pos.push_back(float2 {dis(generator) * imgWidth, dis(generator) * imgWidth});
        vel.push_back(float2 {disVel(generator), disVel(generator)});
    }


    return make_pair(pos, vel);
}

void Particles::generateOnGPU(int n, float imgWidth, float imgHeight, float2* &pos, float2* &vel) {
    checkCudaErrors(cudaMalloc((void**)&pos, n * sizeof(float2)));
    checkCudaErrors(cudaMalloc((void**)&vel, n * sizeof(float2)));
    auto tmp = this->generate(n, imgWidth, imgHeight);
    checkCudaErrors(cudaMemcpy(pos, tmp.first.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(vel, tmp.second.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
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
        int y = floor(p.y);

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
        renderParticles<<<grid, block>>>(Particles::leaderColor,    dLeaderPos,   settings->leaders,   pboData, hMap->glData.imageWidth, hMap->glData.imageHeight);
        renderParticles<<<grid, block>>>(Particles::followerColor,  dFollowerPos, settings->followers, pboData, hMap->glData.imageWidth, hMap->glData.imageHeight);
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
                                activeFollowers, activeLeaders, 2,  //  M, N, K
                                &alpha,
                                dFollowersPosSq, 2,
                                dOnes, 2,
                                &beta,
                                dDistances, activeFollowers);

    //alpha = 1.0f;
    beta = 1.0f;
    response = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           activeFollowers, activeLeaders, 2,  //  M, N, K
                           &alpha,
                           dOnes, 2,
                           dLeadersPosSq, 2,
                           &beta,
                           dDistances, activeFollowers);

    alpha = -2.0f;
    //beta = 1.0f;
    response = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                activeFollowers, activeLeaders, 2,  //  M, N, K
                                &alpha,
                                (float*)dFollowerPos, 2,
                                (float*)dLeaderPos, 2,
                                &beta,
                                dDistances, activeFollowers);
    //checkDeviceMatrix<float>(dDistances,sizeof(float) * activeFollowers, activeLeaders, activeFollowers, "%f ", "M");

}
/*
__device__ __forceinline__ float getDist(float* dDistances, int leader, int follower) {

}
*/
__global__ void moveParticles_Followers(float2* particlePos, float2* particleVel, unsigned int particleCount, float2* leaderPos, unsigned int leaderCount, float* dDistances) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int jump = gridDim.x * blockDim.x;

    float2 *pos = &particlePos[idx];
    float2 *vel = &particleVel[idx];
    while (idx < particleCount) {
        //FIND CLOSEST LEADER
        float *dst = dDistances + idx * leaderCount;
        int clLeaderI = 0;
        float clLeaderDst = *dst;
        for (int i = 1; i < leaderCount; i++, dst++) {
            if (*dst < clLeaderDst) {
                clLeaderDst = *dst;
                clLeaderI = i;
            }
        }
        // MOVE TO LEADER
        float2 clLeaderPos = leaderPos[clLeaderI];
        float2 dir = {(clLeaderPos.x - pos->x) / sqrt(clLeaderDst), (clLeaderPos.y - pos->y) / sqrt(clLeaderDst)};

        *pos = float2{pos->x + dir.x, pos->y + dir.y};


        pos += jump;
        idx += jump;
    }
}

void Particles::moveFollowers() {
    constexpr unsigned int TPB_1D = 128; //TODO -> define somewhere TPB_1D
    dim3 block(TPB_1D, 1, 1);
    dim3 grid((activeLeaders + TPB_1D - 1) / TPB_1D, 1, 1);
    //float2* particlePos, float2* particleVel, unsigned int particleCount, float2* leaderPos, unsigned int leaderCount, float* dDistances
    moveParticles_Followers<<<grid, block>>>(dFollowerPos, dFollowerVel, activeFollowers, dLeaderPos, activeLeaders, dDistances);
}

__global__ void moveParticles_Leaders(float2* particlePos, float2* particleVel, unsigned int particleCount, int imgW, int imgH) {
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int jump = blockDim.x * gridDim.x;

    float2* pos = &particlePos[tx];
    float2* vel = &particleVel[tx];
    while (tx < particleCount) {
        float2 nPos = float2 { pos->x + vel->x, pos->y + vel->y };
        if (nPos.x < 0 || nPos.x >= imgW) {
            vel->x *= -1;
            nPos.x = pos->x + vel->x;
        }
        if (nPos.y < 0 || nPos.y >= imgH) {
            vel->y *= -1;
            nPos.y = pos->y + vel->y;
        }
        *pos = nPos;

        pos += jump;
        vel += jump;
        tx += jump;
    }
}

void Particles::moveLeaders() {
    constexpr unsigned int TPB_1D = 128; //TODO -> define somewhere TPB_1D
    dim3 block(TPB_1D, 1, 1);
    dim3 grid((activeLeaders + TPB_1D - 1) / TPB_1D, 1, 1);
    moveParticles_Leaders<<<grid, block>>>(dLeaderPos, dLeaderVel, activeLeaders, hMap->glData.imageWidth, hMap->glData.imageHeight);
}

void Particles::move() {
    this->moveLeaders();
    this->calculateDistances();
    this->moveFollowers();
}


