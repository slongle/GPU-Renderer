#include "GUI/gui.h"
#include "renderer/pathtracer.h"

#include "renderer/bvh.h"

#include <iostream>
using std::cout;
using std::endl;

int main(int argc, char** argv) {   
    std::string filename = "E:\\Document\\Graphics\\code\\Renderer\\scene\\CornellBox\\CornellBox-Original.obj";
    //filename = "E:\\Document\\Graphics\\code\\Renderer\\scene\\CornellBox\\CornellBox-Sphere.obj";
    //filename = "E:\\Document\\Graphics\\code\\Renderer\\scene\\CornellBox\\CornellBox-Mirror.obj";
    //filename = "E:\\Document\\Graphics\\code\\Renderer\\scene\\CornellBox\\CornellBox-Water.obj";
    //filename = "E:\\Document\\Graphics\\code\\Renderer\\scene\\mori_knob\\testObj.obj";
    //filename = "E:\\Document\\Graphics\\code\\Renderer\\scene\\fireplace_room\\fireplace_room.obj";    
    //filename = "E:\\Document\\Graphics\\code\\Renderer\\scene\\CBox\\cbox(mesh).xml"; 
    filename = "E:\\Document\\Graphics\\code\\Renderer\\scene\\CBox\\cbox(sphere).xml";
    std::shared_ptr<PathTracer> pathTracer(new PathTracer(filename));
      
    
              
    Gui::init(pathTracer);             
    Gui::mainLoop();
    return 0;
} 