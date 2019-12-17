#include "GUI/gui.h"

#include <iostream>
using std::cout;
using std::endl;

#include "renderer/core/scene.h"
#include "renderer/loader/pbrtloader.h"

#include "utility/helper_logger.h"
#include "renderer/core/sampling.h"
#include "renderer/core/cpurender.h"

class F {
public:
    F() {
        x = new int[10];
        for (int i = 0; i < 10; i++) {
            x[i] = i;
        }
    }

    ~F() {
        puts("!");
        delete[] x;
    }

    int* x;
    int a, b, c;
};

F* foo() {
    F* nf = new F();
    return nf;
}

int main() {
    std::string filepath = "E:/Document/Graphics/code/GPU-Renderer/scene/cornell-box/scene.pbrt";    
    filesystem::path path(filepath);
    getFileResolver()->prepend(path.parent_path());
    SceneLoader* sceneLoader = nullptr; 
    sceneLoader = new  PBRTLoader(filepath);
    std::shared_ptr<Renderer> renderer = sceneLoader->Load();



    //render(renderer);
    //return 0;

    Gui::init(renderer);
    Gui::mainLoop();
    return 0;
}