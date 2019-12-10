#include "GUI/gui.h"

#include <iostream>
using std::cout;
using std::endl;

#include "renderer/loader/pbrtloader.h"

#include "utility/helper_logger.h"

int main() {

    std::string filepath = "E:/Document/Graphics/code/GPU-Renderer/scene/cornell-box/scene.pbrt";    
    filesystem::path path(filepath);
    getFileResolver()->prepend(path.parent_path());
    SceneLoader* sceneLoader = nullptr; 
    sceneLoader = new  PBRTLoader(filepath);
    sceneLoader->Load();


    Gui::init();
    Gui::mainLoop();
    return 0;
}