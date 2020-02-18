#include "GUI/gui.h"
#include "renderer/pathtracer.h"

#include "renderer/imageio.h"

#include <iostream>
using std::cout;
using std::endl;

int main(int argc, char** argv) {
    std::string solutionDir("E:/Document/Graphics/code/GPU-Renderer/scene/");
    std::vector<std::string> scenes(100);
    scenes[0] = "CBox/cbox(sphere).xml";
    scenes[1] = "cornell-box/scene.xml";    
    scenes[2] = "veach-bidir/scene.xml";
    scenes[3] = "veach-mis/scene.xml";
    scenes[4] = "veach-ajar/scene.xml";
    scenes[5] = "glass-of-water/scene.xml";
    scenes[6] = "material-testball/scene.xml";
    scenes[7] = "spaceship/scene.xml";
    scenes[8] = "living-room/scene.xml";
    scenes[9] = "staircase/scene.xml";
    const std::string filename(solutionDir + scenes[6]);
    cout << filename << endl;
    std::shared_ptr<PathTracer> pathTracer(new PathTracer(filename));

    //pathTracer->render(1000);

    Gui::init(pathTracer);               
    Gui::mainLoop();
    return 0;
} 