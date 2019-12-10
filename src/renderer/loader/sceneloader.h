#pragma once
#ifndef __SCENELOADER_H
#define __SCENELOADER_H

#include "renderer/core/scene.h"
#include "renderer/core/parameterset.h"

#include <string>
#include <iostream>


class SceneLoader {
public:
    SceneLoader(const std::string& filepath) :m_filepath(filepath) {        
        m_scene = std::make_shared<Scene>();
    }

    ~SceneLoader() = default;

    /**
     * \brief load scene description file and parse it into Scene
     * 
     */
    virtual void Load() = 0;


    filesystem::path m_filepath;
    std::shared_ptr<Scene> m_scene;
};


#endif // __SCENELOADER_H