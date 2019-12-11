#pragma once
#ifndef __PBRTLOADER_H
#define __PBRTLOADER_H

#include "renderer/loader/sceneloader.h"

#include <string_view>

class PBRTLoader : public SceneLoader {
public:

    PBRTLoader(std::string filepath) :SceneLoader(filepath) {}

    // Inherited via SceneLoader
    std::shared_ptr<Renderer> Load() override;
};

class Loc {
public:
    Loc() :m_line(1), m_column(0) {};

    int m_line, m_column;
};

class Tokenizer {
public:
    static std::unique_ptr<Tokenizer> CreateFromFile(
        const std::string filename);

    std::string_view Next();

private:
    Tokenizer(const std::string);

    int GetChar();
    void UngetChar();

    std::string m_content;
    const char* m_pos, * m_end;
    Loc m_loc;
};

#endif // !__PBRTLOADER_H
