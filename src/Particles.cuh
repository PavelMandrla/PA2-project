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

    //DISTANCE CALCULATION DATA
    float* dDistances;
    float* dOnes;
    //LEADER DATA
    float2* dLeaderPos;
    float2* dLeaderDir;
    //FOLLOWER DATA
    float2* dFollowerPos;   // POINTER TO ACTIVE FOLLOWER POSITION ARRAY
    float2* dFollowerPosNext;
    unsigned char* dFollowerStatus;   // INFO, IF THE PARTICLE WAS TERMINATE
    float2* dFollowerPos1;  // TWO FOLLOWER POSITION ARRAYS NEEDED FOR REDUCTION OF TERMINATED PARTICLES
    float2* dFollowerPos2;

    unsigned int activeLeaders;
    unsigned int activeFollowers;

    unsigned int* dActiveFollowersNext;

    cublasStatus_t status;
    cublasHandle_t handle;

    vector<float2> generatePositions(int n);
    vector<float2> generateDirections(int n);
    void generateParticles();

    void moveLeaders(unsigned char* pboData);
    void moveFollowers(unsigned char* pboData);
    void exterminate();
    void calculateDistances();
public:
    shared_ptr<Settings> settings;
    shared_ptr<HeightMap> hMap;

    Particles(shared_ptr<Settings> settings, shared_ptr<HeightMap> hMap);
    ~Particles();

    void move();
};


#endif //PA2PROJECT_PARTICLES_CUH
