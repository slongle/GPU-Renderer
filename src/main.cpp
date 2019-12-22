#include "GUI/gui.h"

#include <iostream>
using std::cout;
using std::endl;

#include "renderer/core/scene.h"
#include "renderer/loader/pbrtloader.h"

#include "utility/helper_logger.h"
#include "renderer/core/sampling.h"
#include "renderer/core/cpurender.h"
#include "renderer/core/gpurender.h"
#include "renderer/core/triangle.h"

int main() {   
    std::vector<std::string> scenes(100);
    scenes[0] = "E:/Document/Graphics/code/GPU-Renderer/scene/cornell-box/scene.pbrt";
    scenes[1] = "E:/Document/Graphics/code/GPU-Renderer/scene/veach-mis/scene.pbrt";
    std::string filepath = scenes[1];    
    filesystem::path path(filepath);
    getFileResolver()->prepend(path.parent_path());
    SceneLoader* sceneLoader = nullptr; 
    sceneLoader = new  PBRTLoader(filepath);
    std::shared_ptr<Renderer> renderer = sceneLoader->Load();  

    //render(renderer);
    //return 0;

    GPURender(renderer);
    return 0;

    Gui::init(renderer);             
    Gui::mainLoop();
    return 0;
}