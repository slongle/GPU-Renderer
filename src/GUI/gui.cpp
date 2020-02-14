// OpenGL Graphics includes
#include "utility/helper_gl.h"
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include "utility/helper_cuda.h"

// Helper functions
#include "utility/helper_cuda.h"
#include "utility/helper_functions.h"
#include "utility/helper_timer.h"

// Header file
#include "gui.h"


int iDivUp(int a, int b) {
    return (a + b - 1) / b;
}

namespace Gui {
    // Window configure
    const char* sample = "CUDA Path Tracing";
    uint width = 512, height = 512;

    // Renderer
    std::shared_ptr<PathTracer> renderer;
    uint32 nIteration = 0;

    // Pixel Buffer
    GLuint pbo = 0;     // OpenGL pixel buffer object
    GLuint tex = 0;     // OpenGL texture object
    struct cudaGraphicsResource* cuda_pbo_resource = nullptr; // CUDA Graphics Resource (to transfer PBO)

    // Timer
    StopWatchInterface* timer = nullptr;

    // FPS
    int fpsCount = 0;        // FPS count for averaging
    int fpsLimit = 1;        // FPS limit for sampling
    unsigned int frameCount = 0;

    // Mouse motion variables
    int ox, oy;
    int buttonState = 0;

    // Stop
    bool start = false;
}

void Gui::init(std::shared_ptr<PathTracer> pathTracer)
{
    // Copy and init renderer
    renderer = pathTracer;
    renderer->init();

    // Copy resolution
    int2 resolution = renderer->getResolution();
    width = resolution.x;
    height = resolution.y;

#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    //start logs
    printf("%s Starting...\n\n", sample);

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    int argc = 0;
    char argv[3][3];

    initGL(&argc, (char **)argv);

    findCudaDevice(argc, (const char**)argv);          // in helper_cuda.h
    sdkCreateTimer(&timer);                            // in helper_timer.h

    // initialize GLUT callback functions
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    // initialize PBO
    initPixelBuffer();

#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif

}

void Gui::mainLoop()
{
    glutMainLoop();
}

void Gui::render()
{    
    // map PBO to get CUDA device pointer
    uint* d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes,
        cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    checkCudaErrors(cudaMemset(d_output, 0, width * height * 4));

    // render
    if (start) {
        renderer->render(d_output);
        nIteration++;
    }

    getLastCudaError("kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

void Gui::computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "Render: %3.1f fps, %u iterations", ifps, nIteration);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(1.f, ifps);
        sdkResetTimer(&timer);
    }
}

void Gui::initPixelBuffer()
{
    if (pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    // create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Gui::initGL(int* argc, char** argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA rendering");

    if (!isGLVersionSupported(2, 0) ||
        !areGLExtensionsSupported("GL_ARB_pixel_buffer_object"))
    {
        printf("Required OpenGL extensions are missing.");
        exit(EXIT_SUCCESS);
    }
}

void Gui::reshape(int w, int h)
{    
    bool state = start;
    start = false;

    width = w;
    height = h;
    initPixelBuffer();

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    renderer->resize(width, height);

    nIteration = 0;
    start = state;
}

void Gui::motion(int x, int y)
{
    bool state = start;
    start = false;

    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 4)
    {
        // right = zoom
        //viewTranslation.z += dy / 100.0f;
        renderer->zoom(dy / 100.0f);
    }
    else if (buttonState == 2)
    {
        // middle = translate
        //viewTranslation.x += dx / 100.0f;
        //viewTranslation.y -= dy / 100.0f;
        renderer->translate(-dx / 100.0f, dy / 100.0f);

    }
    else if (buttonState == 1)
    {
        // left = rotate
        //viewRotation.x += dy / 5.0f;
        //viewRotation.y += dx / 5.0f;
        renderer->rotate(dx / 500.0f, -dy / 500.0f);
    }
    nIteration = 0;

    ox = x;
    oy = y;
    glutPostRedisplay();

    start = state;
}

void Gui::mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        buttonState |= 1 << button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

void Gui::keyboard(unsigned char key, int x, int y)
{
    //bool state = start;
    //start = false;

    //std::cout << (int)key << std::endl;
    switch (key)
    {
    case 13:
        renderer->render(1000);
        break;
    case 27:
#if defined (__APPLE__) || defined(MACOSX)
        exit(EXIT_SUCCESS);
#else
        glutDestroyWindow(glutGetWindow());
        return;
#endif
        break;
    case 32:
        start ^= 1;
        break;
    default:
        break;
    }

    glutPostRedisplay();

    //start = state;
}

void Gui::idle()
{
    glutPostRedisplay();
}

void Gui::display()
{
    sdkStartTimer(&timer);

    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // draw using texture

    // copy from pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
    glutReportErrors();

    sdkStopTimer(&timer);

    computeFPS();
}

void Gui::cleanup()
{
    sdkDeleteTimer(&timer);

    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
    // Calling cudaProfilerStop causes all profile data to be
    // flushed before the application exits
    checkCudaErrors(cudaProfilerStop());
}
