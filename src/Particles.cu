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
    this->dLeaders = this->generateOnGPU(settings->leaders, w, h);
    this->dFollowers = this->generateOnGPU(settings->followers, w, h);

    this->status = cublasStatus_t();
    this->handle = cublasHandle_t();
    this->status = cublasCreate(&handle) ;

    unsigned int onesCount = 2 * (settings->leaders > settings->followers ? settings->leaders : settings->followers);
    cudaMalloc((void**)&this->dOnes, sizeof(float) * onesCount);
    cudaMemcpy(dOnes, vector<float>(onesCount, 1.0f).data(), sizeof(float) * onesCount, cudaMemcpyHostToDevice);

    //cudaMalloc( (void**)&dDistances, settings->leaders * settings->followers * sizeof(float));
}

vector<Particle> Particles::generate(int n, float imgWidth, float imgHeight) {
    std::mt19937 generator(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    std::vector<Particle> result;

    std::uniform_real_distribution<float> dis(0.0, 1.0);

    float dW = imgWidth / float(n);
    float dH = imgHeight / float(n);
    for (int i = 0; i < n; i++) {
        result.push_back(Particle {
                i * dW,
                i * dH,
                0.0f, 0.0f
        });

        /*
        result.push_back(Particle {
                dis(generator) * imgWidth,
                dis(generator) * imgHeight,
                0.0f, 0.0f
        });*/
    }
    return result;
}

Particle* Particles::generateOnGPU(int n, float imgWidth, float imgHeight) {
    Particle* result;
    auto particles = Particles::generate(n, imgWidth, imgHeight);
    checkCudaErrors(cudaMalloc((void**)&result, n * sizeof(Particle)));
    checkCudaErrors(cudaMemcpy(result, particles.data(), n * sizeof(Particle), cudaMemcpyHostToDevice));
    return result;
}

Particles::~Particles() {
    status = cublasDestroy(handle);
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
        renderParticles<<<grid, block>>>(Particles::leaderColor,    dLeaders,   settings->leaders,   pboData, hMap->glData.imageWidth, hMap->glData.imageHeight);
        renderParticles<<<grid, block>>>(Particles::followerColor,  dFollowers, settings->followers, pboData, hMap->glData.imageWidth, hMap->glData.imageHeight);
    };

    checkCudaErrors(cudaGraphicsUnmapResources(1, &hMap->cudaData.pboResource, 0));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, hMap->glData.pboID);
    auto err = glGetError();
    glBindTexture(GL_TEXTURE_2D, hMap->overlayTexId);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, hMap->glData.imageWidth, hMap->glData.imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);   //Source parameter is NULL, Data is coming from a PBO, not host memory
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

__global__ void createSquareMatrix(Particle* particles, int particleCount, float* dst) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int jump = gridDim.x * blockDim.x;

    Particle* particle = &particles[idx];
    float* row = &dst[2*idx];
    while (idx < particleCount) {
        row[0] = particle->x * particle->x;
        row[1] = particle->y * particle->y;

        particle += jump;
        row += jump;
        idx += jump;
    }
}

inline void squarePositions(float* &dPositionsSq, Particle* dParticles, unsigned int activeParticles) {
    checkCudaErrors(cudaMalloc((void**)&dPositionsSq, 2 * activeParticles * sizeof(float)));
    constexpr unsigned int TPB_1D = 128;
    dim3 block(128,1,1); //TODO -> change TPB1D?
    dim3 grid((activeParticles + TPB_1D - 1) / TPB_1D);
    createSquareMatrix<<<block, grid>>>(dParticles, activeParticles, dPositionsSq);
}

void Particles::calculateDistances() {
    // M*N distance matrix
    float* dDistances;
    checkCudaErrors(cudaMalloc( (void**)&dDistances, activeLeaders * activeFollowers * sizeof(float)));

    // SQUARE OF POSITIONS
    float* dLeadersPos;
    float* dFollowersPos;
    float* dLeadersPosSq;
    float* dFollowersPosSq;
    squarePositions(dLeadersPosSq, dLeaders, activeLeaders);
    squarePositions(dFollowersPosSq, dFollowers, activeFollowers);

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
                           dOnes, 2,
                           dLeadersPosSq, 2,
                           &beta,
                           dDistances, activeFollowers);



    //checkDeviceMatrix<float>(dDistances,	sizeof(float)*activeLeaders, activeFollowers, activeLeaders, "%f ", "M");
    checkDeviceMatrix<float>(dFollowersPosSq, sizeof(float) * 2, activeFollowers, 2, "%f ", "Leaders pos square");
    //checkDeviceMatrix<float>(dLeadersPosSq, sizeof(float)*2, activeLeaders, 2, "%f ", "Leaders pos square");
    checkDeviceMatrix<float>(dOnes, sizeof(float)*2, activeLeaders, 2, "%f ", "Leaders pos square");
    checkDeviceMatrix<float>(dDistances,	sizeof(float)*activeFollowers,activeLeaders, activeFollowers, "%f ", "M");
    //checkDeviceMatrix<float>(dFollowersPosSq, sizeof(float)*2, activeFollowers, 2, "%f ", "Leaders pos square");

}




