#ifndef __GUI_H
#define __GUI_H

#include <cuda_runtime.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif


// CUDA functions in kernel.cu
extern "C" void freeCudaBuffers();
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint * d_output, uint imageW, uint imageH);
extern "C" void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix);

namespace Gui {

    /**
     * \brief Initialize GUI
     *
     */
    void init();

    /**
     * \brief Main loop of GUI
     *
     */
    void mainLoop();

    /**
     * \brief Invoke CUDA render function
     */
    void render();

    /**
     * \brief Compute FPS     
     */
    void computeFPS();

    /**
     * \brief Initialize Pixel Buffer Object and Texture
     *
     */
    void initPixelBuffer();

    /**
     * \brief Initialize OpenGL
     *
     */
    void initGL(int* argc, char** argv);

//**************************************************
//          
//  Callback Functions
//
//**************************************************

    void reshape(int w, int h);
    void motion(int x, int y);
    void mouse(int button, int state, int x, int y);
    void keyboard(unsigned char key, int x, int y);
    void idle();
    void display();
    void cleanup();    

};

#endif // !__GUI_H
