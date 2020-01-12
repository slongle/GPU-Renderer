#include "GUI/gui.h"

#include <iostream>
using std::cout;
using std::endl;

#include "renderer/loader/pbrtloader.h"
#include "renderer/core/cpurender.h"
#include "renderer/core/gpurender.h"

#include "utility/helper_timer.h"
StopWatchInterface* timer = nullptr;

int main(int argc, char** argv) {   
    int idx;
    do {
        cout << "Please input scene id : 0/1\n";
        std::cin >> idx;
    } while (idx != 0 && idx != 1);

    std::vector<std::string> scenes(100);
    //scenes[0] = "E:/Document/Graphics/code/GPU-Renderer/scene/cornell-box/scene-diffuse.pbrt";
    //scenes[1] = "E:/Document/Graphics/code/GPU-Renderer/scene/cornell-box/scene-glass.pbrt";
    scenes[0] = "scene/cornell-box/scene-diffuse.pbrt";
    scenes[1] = "scene/cornell-box/scene-glass.pbrt";
    /*scenes[1] = "E:/Document/Graphics/code/GPU-Renderer/scene/veach-mis/scene.pbrt";
    scenes[2] = "E:/Document/Graphics/code/GPU-Renderer/scene/veach-bidir/scene.pbrt";
    scenes[3] = "E:/Document/Graphics/code/GPU-Renderer/scene/caustic-glass/scene.pbrt";
    scenes[4] = "E:/Document/Graphics/code/GPU-Renderer/scene/water-caustic/scene.pbrt";*/

    std::string filepath = scenes[idx];
    filesystem::path path(filepath);
    getFileResolver()->prepend(path.parent_path());
    SceneLoader* sceneLoader = nullptr; 
    sceneLoader = new  PBRTLoader(filepath);
    std::shared_ptr<Renderer> renderer = sceneLoader->Load();  

    float t;

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    render(renderer);
    sdkStopTimer(&timer);
    t = sdkGetAverageTimerValue(&timer) / 1000.f;
    printf("%f s\n", t);


    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    gpu_render(renderer);
    sdkStopTimer(&timer);
    t = sdkGetAverageTimerValue(&timer) / 1000.f;
    printf("%f s\n", t);

    return 0;

    Gui::init(renderer);             
    Gui::mainLoop();
    sdkDeleteTimer(&timer);
    return 0;
}