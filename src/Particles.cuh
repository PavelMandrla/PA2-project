#ifndef PA2PROJECT_PARTICLES_CUH
#define PA2PROJECT_PARTICLES_CUH

#include <vector>
#include <memory>
#include "HeightMap.cuh"

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
    vector<Particle> generate(int n, float imgWidth, float imgHeight);
    Particle* generateOnGPU(int n, float imgWidth, float imgHeight);
public:
    int leaderCount;
    int followerCount;
    shared_ptr<HeightMap> hMap;
    Particle* dLeaders;
    Particle* dFollowers;

    Particles(int lCount, int fCount, shared_ptr<HeightMap> hMap);
    ~Particles();

    void renderToOverlay();
};


#endif //PA2PROJECT_PARTICLES_CUH
