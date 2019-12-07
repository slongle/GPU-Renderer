#pragma once
#ifndef __MITSUBALOADER_H
#define __MITSUBALOADER_H 

#include "sceneloader.h"

class MitsubaLoader : public SceneLoader {
public:
    MitsubaLoader(std::string filepath) :SceneLoader(filepath) {}

    // Inherited via SceneLoader
    void load();

private:
    std::string getOffset(ptrdiff_t pos) const;

};

#endif // __MITSUBALOADER_H