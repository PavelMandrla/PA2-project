#include "Particles.cuh"

#include <random>
#include <chrono>
#include <cudaDefs.h>

Particles::Particles(int lCount, int fCount, shared_ptr<HeightMap> hMap) {
    this->leaderCount = lCount;
    this->followerCount = fCount;
    this->hMap = hMap;

    float w = float(hMap->glData.imageWidth);
    float h = float(hMap->glData.imageHeight);
    this->dLeaders = this->generateOnGPU(lCount, w, h);
    this->dFollowers = this->generateOnGPU(fCount, w, h);
}

vector<Particle> Particles::generate(int n, float imgWidth, float imgHeight) {
    std::mt19937 generator(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    std::vector<Particle> result;

    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < n; i++) {
        result.push_back(Particle {
                dis(generator) * imgWidth,
                dis(generator) * imgHeight,
                0.0f, 0.0f
        });
    }
    return result;
}

Particle* Particles::generateOnGPU(int n, float imgWidth, float imgHeight) {
    Particle* result;
    auto particles = Particles::generate(n, imgWidth, imgHeight);
    checkCudaErrors(cudaMalloc((void**)&result, n * sizeof(Particles)));
    checkCudaErrors(cudaMemcpy(result, particles.data(), n * sizeof(Particles), cudaMemcpyHostToDevice));
    return result;
}

Particles::~Particles() {
    if (this->dLeaders) cudaFree(this->dLeaders);
    if (this->dFollowers) cudaFree(this->dFollowers);
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

__global__ void renderParticles(uchar3 color, Particle* particles, int particleCount, unsigned char* pbo, const unsigned int pboWidth, const unsigned int pboHeight) {
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int jump = blockDim.x * gridDim.x;

    while (tx < particleCount) {
        Particle p = particles[tx];
        unsigned int pboIdx = ((floor(p.y) * pboWidth) + floor(p.x)) * 4;
        pbo[pboIdx++] = color.x;
        pbo[pboIdx++] = color.y;
        pbo[pboIdx++] = color.z;
        pbo[pboIdx]   = 255;

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
        renderParticles<<<grid, block>>>(Particles::leaderColor,    dLeaders,   this->leaderCount,   pboData, hMap->glData.imageWidth, hMap->glData.imageHeight);
        renderParticles<<<grid, block>>>(Particles::followerColor,  dFollowers, this->followerCount, pboData, hMap->glData.imageWidth, hMap->glData.imageHeight);
    };

    checkCudaErrors(cudaGraphicsUnmapResources(1, &hMap->cudaData.pboResource, 0));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, hMap->glData.pboID);
    glBindTexture(GL_TEXTURE_2D, hMap->overlayTexId);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, hMap->glData.imageWidth, hMap->glData.imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);   //Source parameter is NULL, Data is coming from a PBO, not host memory
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

}


