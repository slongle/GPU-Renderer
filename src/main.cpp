#include "GUI/gui.h"

#include <iostream>
using std::cout;
using std::endl;

#include "renderer/core/fwd.h"
#include "renderer/loader/mitsubaloader.h"

#include "utility/helper_logger.h"

int main() {

    std::string filepath = "E:/Document/Graphics/code/GPU-Renderer/scene/volumetric-caustic/scene.xml";    
    filesystem::path path(filepath);
    getFileResolver()->prepend(path.parent_path());
    SceneLoader* sceneLoader = nullptr; 
    sceneLoader = new  MitsubaLoader(filepath);
    sceneLoader->load();


    //Gui::init();
    //Gui::mainLoop();
    return 0;
}