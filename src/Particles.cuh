#ifndef PA2PROJECT_PARTICLES_CUH
#define PA2PROJECT_PARTICLES_CUH

#include <vector>
#include <memory>
#include <utility>
#include "HeightMap.cuh"

#include <cudaDefs.h>
#include <cublas_v2.h>

using namespace std;

class Particles {
private:
    static constexpr uchar3 leaderColor = {255, 0, 0};
    static constexpr uchar3 followerColor = {0, 0, 255};

    unsigned int activeLeaders;
    unsigned int activeFollowers;

    cublasStatus_t status;
    cublasHandle_t handle;

    float* dOnes;

    pair<vector<float2>, vector<float2>> generate(int n, float imgWidth, float imgHeight);
    void generateOnGPU(int n, float imgWidth, float imgHeight, float2* &pos, float2* &vel);

    void moveLeaders();
    void moveFollowers();
    void calculateDistances();
public:
    shared_ptr<Settings> settings;
    shared_ptr<HeightMap> hMap;

    //LEADER DATA
    float2* dLeaderPos;
    float2* dLeaderVel;
    //FOLLOWER DATA
    float2* dFollowerPos;
    float2* dFollowerVel;

    float* dDistances;

    Particles(shared_ptr<Settings> settings, shared_ptr<HeightMap> hMap);
    ~Particles();

    void renderToOverlay();
    void move();
};


#endif //PA2PROJECT_PARTICLES_CUH
