#include "Particles.cuh"

#include <random>
#include <chrono>
#include <cudaDefs.h>

void Particles::init(int lCount, int fCount, unsigned int imgWidth, unsigned int imgHeight) {
    this->leaderCount = lCount;
    this->followerCount = fCount;

    float w = float(imgWidth);
    float h = float(imgHeight);
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

void Particles::renderToTexture() {

}


