#pragma once
#ifndef __SPECTRUM_H
#define __SPECTRUM_H

#include "renderer/core/fwd.h"

class Spectrum {
public:
    Spectrum(Float r = 0, Float g = 0, Float b = 0) :r(r), g(g), b(b) {}

    Float operator[] (int idx) const;

    Float r, g, b;


    friend 
    void SpectrumToUnsignedChar(
        const Spectrum& s, 
        unsigned char* const uc, 
        int len = 3);
};

#endif // !__SPECTRUM_H
