#ifndef PA2PROJECT_PARTICLES_CUH
#define PA2PROJECT_PARTICLES_CUH

#include <vector>
#include <memory>
#include "HeightMap.cuh"

#include <cudaDefs.h>
#include <cublas_v2.h>

using namespace std;

struct Particle {
    float x, y;         // POSITION
    float v_x { 0.0f }; // VELOCITY IN DIRECTION X
    float v_y { 0.0f }; // VELOCITY IN DIRECTION Y
};

class Particles {
private:
    static constexpr uchar3 leaderColor = {255, 0, 0};
    static constexpr uchar3 followerColor = {0, 0, 255};

    unsigned int activeLeaders;
    unsigned int activeFollowers;

    cublasStatus_t status;
    cublasHandle_t handle;

    float* dOnes;

    vector<Particle> generate(int n, float imgWidth, float imgHeight);
    Particle* generateOnGPU(int n, float imgWidth, float imgHeight);
public:
    shared_ptr<Settings> settings;
    shared_ptr<HeightMap> hMap;
    Particle* dLeaders;
    Particle* dFollowers;

    Particles(shared_ptr<Settings> settings, shared_ptr<HeightMap> hMap);
    ~Particles();

    void renderToOverlay();
    void calculateDistances();
};


#endif //PA2PROJECT_PARTICLES_CUH
