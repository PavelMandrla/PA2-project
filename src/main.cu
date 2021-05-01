#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include <cudaDefs.h>

#include "Settings.h"
#include "HeightMap.cuh"
#include "Particles.cuh"

using namespace std;

//TODO -> update TPB_1D and TPB_2D values
constexpr unsigned int TPB_1D = 8;									// ThreadsPerBlock in one dimension
constexpr unsigned int TPB_2D =  TPB_1D * TPB_1D;					// ThreadsPerBlock = TPB_1D*TPB_1D (2D block)

//cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();
Settings settings;
HeightMap hMap;
Particles particles;


#pragma region --- CUDA ---

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
/*
void cudaWorker() {
    // Map GL resources
    checkCudaErrors(cudaGraphicsMapResources(1, &hMap.cudaData.texResource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&hMap.cudaData.texArrayData, hMap.cudaData.texResource, 0, 0));

    // TODO -> move pbo resource to be part of overlay texture
    checkCudaErrors(cudaGraphicsMapResources(1, &hMap.cudaData.pboResource, 0));
    unsigned char* pboData;
    size_t pboSize;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&pboData, &pboSize, hMap.cudaData.pboResource));

    {   // CLEAR PBO
        dim3 block(TPB_1D, TPB_1D, 1);
        dim3 grid((hMap.glData.imageWidth + TPB_1D - 1) / TPB_1D, (hMap.glData.imageHeight + TPB_1D - 1) / TPB_1D, 1);
        clearPBO<<<grid, block>>>(pboData, hMap.glData.imageWidth, hMap.glData.imageHeight);
    };

    {   // PUT PARTCLES INTO PBO
        constexpr uchar3 leaderColor = {255, 0, 0};
        constexpr uchar3 followerColor = {0, 0, 255};

        //TODO -> adjust block and grid sizes
        dim3 block(128, 1, 1);
        dim3 grid(1, 1, 1);
        renderParticles<<<grid, block>>>(leaderColor, dLeaders, settings.leaders, pboData, hMap.glData.imageWidth, hMap.glData.imageHeight);
        renderParticles<<<grid, block>>>(followerColor, dFollowers, settings.followers, pboData, hMap.glData.imageWidth, hMap.glData.imageHeight);
    };

    // TODO -> Run kernel


    // Unmap GL Resources
    checkCudaErrors(cudaGraphicsUnmapResources(1, &hMap.cudaData.texResource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &hMap.cudaData.pboResource, 0));

    // This updates GL texture from PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, hMap.glData.pboID);
    glBindTexture(GL_TEXTURE_2D, hMap.overlayTexId);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, hMap.glData.imageWidth, hMap.glData.imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);   //Source parameter is NULL, Data is coming from a PBO, not host memory
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}
*/
#pragma endregion

#pragma region --- OPEN_GL ---

void my_idle() {
    //cudaWorker();
    glutPostRedisplay();
}

void initGL(int argc, char** argv) {
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(hMap.glData.viewportWidth, hMap.glData.viewportHeight);
    glutInitWindowPosition(0, 0);
    glutSetOption(GLUT_RENDERING_CONTEXT, false ? GLUT_USE_CURRENT_CONTEXT : GLUT_CREATE_NEW_CONTEXT);
    glutCreateWindow(0);

    glutSetWindowTitle("Parallel Papplications 2 - project [MAN0117]");

    glutDisplayFunc([](){ hMap.display(); });
    glutReshapeFunc([](GLsizei w, GLsizei h){ hMap.resize(w, h); });
    glutIdleFunc(my_idle);
    glutSetCursor(GLUT_CURSOR_CROSSHAIR);

    // initialize necessary OpenGL extensions
    glewInit();

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glShadeModel(GL_SMOOTH);
    glViewport(0, 0, hMap.glData.viewportWidth, hMap.glData.viewportHeight);

    glFlush();
}

#pragma endregion

int main(int argc, char* argv[]) {
    #pragma region initialize
    initializeCUDA(deviceProp);
    if (argc < 2) {
        printf("Please specify path to the configuration path");
        return 1;
    }
    settings.init(argv[1]);
    initGL(1, argv);
    // INITIALIZE HEIHGHT MAP
    hMap.init(settings.heightMap);
    // CREATE LEADERS AND FOLLOWERS ON DEVICE
    particles.init(settings.leaders, settings.followers, hMap.glData.imageWidth, hMap.glData.imageHeight);

    #pragma endregion

    glutMainLoop();

    return 0;
}