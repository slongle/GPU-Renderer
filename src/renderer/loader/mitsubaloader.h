#pragma once
#ifndef __MITSUBALOADER_H
#define __MITSUBALOADER_H 

#include "renderer/loader/sceneloader.h"

class MitsubaLoader : public SceneLoader {
public:
    MitsubaLoader(std::string filepath) :SceneLoader(filepath) {}

    // Inherited via SceneLoader
    std::shared_ptr<Renderer> Load() override;

private:
    std::string getOffset(ptrdiff_t pos) const;

};

#endif // __MITSUBALOADER_H