#ifndef PA2PROJECT_PARTICLES_CUH
#define PA2PROJECT_PARTICLES_CUH

#include <vector>

using namespace std;

struct Particle {
    float x, y;         // POSITION
    float v_x { 0.0f }; // VELOCITY IN DIRECTION X
    float v_y { 0.0f }; // VELOCITY IN DIRECTION Y
};

class Particles {
private:
    vector<Particle> generate(int n, float imgWidth, float imgHeight);
    Particle* generateOnGPU(int n, float imgWidth, float imgHeight);
public:
    int leaderCount;
    int followerCount;
    Particle* dLeaders;
    Particle* dFollowers;

    Particles() = default;
    ~Particles();

    void init(int lCount, int fCount, unsigned int imgWidth, unsigned int imgHeight);
    void renderToTexture();
};


#endif //PA2PROJECT_PARTICLES_CUH
