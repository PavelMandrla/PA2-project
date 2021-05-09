#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cudaDefs.h>
#include <memory>
#include "Settings.h"
#include "HeightMap.cuh"
#include "Particles.cuh"

using namespace std;

cudaDeviceProp deviceProp = cudaDeviceProp();
shared_ptr<Settings> settings;
shared_ptr<HeightMap> hMap;
shared_ptr<Particles> particles;


void initGL(int argc, char** argv) {
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(settings->viewportWidth, settings->viewportHeight);
    glutInitWindowPosition(0, 0);
    glutSetOption(GLUT_RENDERING_CONTEXT, false ? GLUT_USE_CURRENT_CONTEXT : GLUT_CREATE_NEW_CONTEXT);
    glutCreateWindow(0);

    glutSetWindowTitle("Parallel Papplications 2 - project [MAN0117]");

    glutDisplayFunc([](){ hMap->display(); });
    glutReshapeFunc([](GLsizei w, GLsizei h){ hMap->resize(w, h); });
    glutIdleFunc([](){
        particles->move();
        glutPostRedisplay();
    });
    glutSetCursor(GLUT_CURSOR_CROSSHAIR);

    // initialize necessary OpenGL extensions
    glewInit();

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glShadeModel(GL_SMOOTH);
    glViewport(0, 0, settings->viewportWidth, settings->viewportHeight);

    glFlush();
}

int main(int argc, char* argv[]) {
    initializeCUDA(deviceProp);
    if (argc < 2) {
        printf("Please specify path to the configuration path");
        return 1;
    }
    settings = make_shared<Settings>(argv[1]);
    initGL(1, argv);
    hMap = make_shared<HeightMap>(settings);
    particles = make_shared<Particles>(settings, hMap);

    glutMainLoop();

    return 0;
}