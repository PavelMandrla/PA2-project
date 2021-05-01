#ifndef PA2PROJECT_HEIGHMAP_CUH
#define PA2PROJECT_HEIGHMAP_CUH

#include <GL/glew.h>
#include <memory>
#include <string>
#include "Settings.h"

using namespace std;

struct GLData {
    unsigned int imageWidth;
    unsigned int imageHeight;
    unsigned int imageBPP;
    unsigned int imagePitch;

    unsigned int pboID;
    unsigned int textureID;
    unsigned int viewportWidth = 1024;
    unsigned int viewportHeight = 1024;
};

struct CudaData {
    cudaTextureDesc			texDesc;				// Texture descriptor used to describe texture parameters

    cudaArray_t				texArrayData;			// Source texture data
    cudaResourceDesc		resDesc;				// A resource descriptor for obtaining the texture data
    cudaChannelFormatDesc	texChannelDesc;			// Texture channel descriptor to define channel bytes
    cudaTextureObject_t		texObj;					// Cuda Texture Object to be produces

    cudaGraphicsResource_t  texResource;
    cudaGraphicsResource_t	pboResource;

    CudaData() {
        memset(this, 0, sizeof(CudaData));			// DO NOT DELETE THIS !!!
    }
};

class HeightMap {
private:
    void prepareGlObjects(const char* imageFileName);
    void initCUDAObjects();
    void initOverlayTexture();
public:
    GLData glData;
    CudaData cudaData;
    unsigned int overlayTexId;

    HeightMap(shared_ptr<Settings> settings);
    ~HeightMap();

    void display();
    void resize(GLsizei w, GLsizei h);
};


#endif //PA2PROJECT_HEIGHMAP_CUH
