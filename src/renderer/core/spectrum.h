#pragma once
#ifndef __SPECTRUM_H
#define __SPECTRUM_H

#include "renderer/core/fwd.h"

class Spectrum {
public:
    __device__ __host__ Spectrum(Float r = 0, Float g = 0, Float b = 0) :r(r), g(g), b(b) {}
    __device__ __host__ Spectrum(const Normal3f& n) : r(std::fabs(n.x)), g(std::fabs(n.y)), b(std::fabs(n.z)) {}

    Float operator[] (int idx) const;
    Spectrum& operator += (const Spectrum& s);

    Float r, g, b;


    friend __device__ __host__
    void SpectrumToUnsignedChar(
        const Spectrum& s, 
        unsigned char* const uc, 
        int len = 3);
};

inline __device__ __host__
Float Clamp(Float a) {
    if (a > 1) return 1;
    else if (a < 0) return 0;
    else return a;
}

inline __device__ __host__
Float Spectrum::operator[](int idx) const
{
    if (idx == 0) return r;
    else if (idx == 1) return g;
    else return b;
}

inline __device__ __host__
Spectrum& Spectrum::operator+=(const Spectrum& s)
{
    r += s.r;
    g += s.g;
    b += s.b;
    return *this;
}

inline __device__ __host__
void SpectrumToUnsignedChar(
    const Spectrum& s,
    unsigned char* const uc,
    int len)
{
    for (int i = 0; i < 3; i++) {
        uc[i] = (unsigned char)(Clamp(s[i]) * 255);
    }
    if (len == 4) {
        uc[3] = 255;
    }
}


#endif // !__SPECTRUM_H
