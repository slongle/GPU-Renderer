#include "GUI/gui.h"

#include <iostream>
using std::cout;
using std::endl;

#include "renderer/loader/pbrtloader.h"
#include "renderer/core/cpurender.h"
#include "renderer/core/gpurender.h"

int main() {   
    std::vector<std::string> scenes(100);
    scenes[0] = "E:/Document/Graphics/code/GPU-Renderer/scene/cornell-box/scene.pbrt";
    scenes[1] = "E:/Document/Graphics/code/GPU-Renderer/scene/veach-mis/scene.pbrt";
    scenes[2] = "E:/Document/Graphics/code/GPU-Renderer/scene/veach-bidir/scene.pbrt";
    std::string filepath = scenes[2];
    filesystem::path path(filepath);
    getFileResolver()->prepend(path.parent_path());
    SceneLoader* sceneLoader = nullptr; 
    sceneLoader = new  PBRTLoader(filepath);
    std::shared_ptr<Renderer> renderer = sceneLoader->Load();  

    render(renderer);
    return 0;

    Gui::init(renderer);             
    Gui::mainLoop();
    return 0;
}