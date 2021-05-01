#include "HeightMap.cuh"

#include <cudaDefs.h>
#include <cuda_gl_interop.h>
#include <GL/freeglut.h>
#include <GL/glew.h>
#include <imageManager.h>
#include <vector>

HeightMap::HeightMap(shared_ptr<Settings> settings) {
    prepareGlObjects(settings->heightMap.c_str());
    this->glData.viewportWidth = settings->viewportWidth;
    this->glData.viewportHeight = settings->viewportHeight;
    initCUDAObjects();
    initOverlayTexture();
}

HeightMap::~HeightMap() {
    checkCudaErrors(cudaGraphicsUnregisterResource(this->cudaData.texResource));
    checkCudaErrors(cudaGraphicsUnregisterResource(this->cudaData.pboResource));

    if (this->glData.textureID > 0)
        glDeleteTextures(1, &this->glData.textureID);
    if (this->glData.pboID > 0)
        glDeleteBuffers(1, &this->glData.pboID);
}

void HeightMap::prepareGlObjects(const char *imageFileName) {
    FIBITMAP* tmp = ImageManager::GenericLoader(imageFileName, 0);
    glData.imageWidth = FreeImage_GetWidth(tmp);
    glData.imageHeight = FreeImage_GetHeight(tmp);
    glData.imageBPP = FreeImage_GetBPP(tmp);
    glData.imagePitch = FreeImage_GetPitch(tmp);

    //OpenGL Texture
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &glData.textureID);
    glBindTexture(GL_TEXTURE_2D, glData.textureID);

    //WARNING: Just some of inner format are supported by CUDA!!!
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, glData.imageWidth, glData.imageHeight, 0, GL_RED, GL_UNSIGNED_BYTE, FreeImage_GetBits(tmp));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);


    glBindTexture(GL_TEXTURE_2D, 0);
    FreeImage_Unload(tmp);

    glGenBuffers(1, &glData.pboID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glData.pboID);														// Make this the current UNPACK buffer (OpenGL is state-based)
    glBufferData(GL_PIXEL_UNPACK_BUFFER, glData.imageWidth * glData.imageHeight * 4, NULL, GL_DYNAMIC_COPY);	// Allocate data for the buffer. 4-channel 8-bit image
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void HeightMap::initCUDAObjects() {
    // Register Image to cuda tex resource
    checkCudaErrors(cudaGraphicsGLRegisterImage(
            &cudaData.texResource,
            glData.textureID,
            GL_TEXTURE_2D,
            cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsReadOnly
    ));

    // Map reousrce and retrieve pointer to undelying array data
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaData.texResource, 0)); //OPENGL, pls nepracuj ted s tou texturou
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cudaData.texArrayData, cudaData.texResource, 0, 0));    //z resourcu chci tahat pixelova data

    // Set resource descriptor
    cudaData.resDesc.resType = cudaResourceType::cudaResourceTypeArray;
    cudaData.resDesc.res.array.array = cudaData.texArrayData;

    // Set Texture Descriptor: Tex Units will know how to read the texture
    cudaData.texDesc.readMode = cudaReadModeElementType;
    cudaData.texDesc.normalizedCoords = false;
    cudaData.texDesc.filterMode = cudaFilterModePoint;
    cudaData.texDesc.addressMode[0] = cudaAddressModeClamp;
    cudaData.texDesc.addressMode[1] = cudaAddressModeClamp;

    // Set Channel Descriptor: How to interpret individual bytes
    checkCudaErrors(cudaGetChannelDesc(&cudaData.texChannelDesc, cudaData.texArrayData));

    // Create CUDA Texture Object
    checkCudaErrors(cudaCreateTextureObject(&cudaData.texObj, &cudaData.resDesc, &cudaData.texDesc, nullptr));

    // Unmap resource: Release the resource for OpenGL
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaData.texResource, 0));

    // Register PBO
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(
            &cudaData.pboResource,
            glData.pboID,
            cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsWriteDiscard
    ));
}

void HeightMap::initOverlayTexture() {
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &overlayTexId);
    glBindTexture(GL_TEXTURE_2D, overlayTexId);

    std::vector<GLubyte> emptyData(this->glData.imageWidth * this->glData.imageHeight * 4, 200);
    glTexImage2D(GL_TEXTURE_2D,0, GL_RGBA, this->glData.imageWidth, this->glData.imageHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, &emptyData[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void HeightMap::display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);

    //glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->glData.textureID);
    glBegin(GL_QUADS);
    glTexCoord2d(0, 0);		glVertex2d(0, 0);
    glTexCoord2d(1, 0);		glVertex2d(this->glData.viewportWidth, 0);
    glTexCoord2d(1, 1);		glVertex2d(this->glData.viewportWidth, this->glData.viewportHeight);
    glTexCoord2d(0, 1);		glVertex2d(0, this->glData.viewportHeight);
    glEnd();

    //glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, this->overlayTexId);
    glBegin(GL_QUADS);
    glTexCoord2d(0, 0);		glVertex2d(0, 0);
    glTexCoord2d(1, 0);		glVertex2d(this->glData.viewportWidth, 0);
    glTexCoord2d(1, 1);		glVertex2d(this->glData.viewportWidth, this->glData.viewportHeight);
    glTexCoord2d(0, 1);		glVertex2d(0, this->glData.viewportHeight);
    glEnd();

    glDisable(GL_TEXTURE_2D);

    glFlush();
    glutSwapBuffers();
}

void HeightMap::resize(GLsizei w, GLsizei h) {
    this->glData.viewportWidth = w;
    this->glData.viewportHeight = h;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, this->glData.viewportWidth, this->glData.viewportHeight);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, this->glData.viewportWidth, 0, this->glData.viewportHeight);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glutPostRedisplay();
}


