#ifndef __GUI_H
#define __GUI_H

#include <cuda_runtime.h>

#include "renderer/pathtracer.h"

typedef unsigned int uint;
typedef unsigned char uchar;
 
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

namespace Gui {

    /**
     * \brief Initialize GUI
     *
     */
    void init(std::shared_ptr<PathTracer> pathTracer);

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
