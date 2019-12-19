#pragma once
#ifndef __SPECTRUM_H
#define __SPECTRUM_H

#include "renderer/core/fwd.h"

class Spectrum {
public:
    __device__ __host__ Spectrum(Float v = 0) :r(v), g(v), b(v) {}
    __device__ __host__ Spectrum(Float r, Float g, Float b) :r(r), g(g), b(b) {}
    __device__ __host__ Spectrum(const Normal3f& n) : r(std::fabs(n.x)), g(std::fabs(n.y)), b(std::fabs(n.z)) {}
    __device__ __host__ Spectrum(const std::vector<Float>& v);

    __device__ __host__ Float operator[] (int idx) const;
    __device__ __host__ Spectrum& operator += (const Spectrum& s);
    __device__ __host__ Spectrum operator * (const Spectrum& s) const;
    __device__ __host__ Spectrum& operator *= (const Spectrum& s);
    __device__ __host__ Spectrum operator / (const Float v) const;
    __device__ __host__ Spectrum& operator /= (const Float v);

    __device__ __host__ Float Max() const;
    __device__ __host__ bool isBlack() const;

    Float r, g, b;


    friend __device__ __host__
    void SpectrumToUnsignedChar(
        const Spectrum& s, 
        unsigned char* const uc, 
        int len = 3);
};

inline __device__ __host__
Float Clamp(Float a,Float l,Float r) {
    if (a > r) return r;
    else if (a < l) return l;
    else return a;
}

inline __device__ __host__
Spectrum::Spectrum(const std::vector<Float>& v)
{
    ASSERT(v.size() == 3, "Spectrum");
    r = v[0];
    g = v[1];
    b = v[2];
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
Spectrum Spectrum::operator*(const Spectrum& s) const
{
    return Spectrum(r * s.r, g * s.g, b * s.b);
}

inline __device__ __host__ 
Spectrum& Spectrum::operator*=(const Spectrum& s)
{
    r *= s.r;
    g *= s.g;
    b *= s.b;
    return *this;
}

inline __device__ __host__ 
Spectrum Spectrum::operator/(const Float v) const
{
    ASSERT(v != 0, "Divide zero");
    Float invV = 1 / v;
    return Spectrum(r * invV, g * invV, b * invV);
}

inline __device__ __host__ 
Spectrum& Spectrum::operator/=(const Float v)
{
    ASSERT(v != 0, "Divide zero");
    Float invV = 1 / v;
    r *= invV;
    g *= invV;
    b *= invV;
    return *this;
}

inline __device__ __host__ 
Float Spectrum::Max() const
{
    return max(max(r, g), b);
}

inline __device__ __host__ 
bool Spectrum::isBlack() const
{
    return r == 0 && g == 0 && b == 0;
}

inline __device__ __host__
void SpectrumToUnsignedChar(
    const Spectrum& s,
    unsigned char* const uc,
    int len)
{
#define TO_BYTE(v) (uint8_t) Clamp(255.f * GammaCorrect(v) + 0.5f, 0.f, 255.f)
    for (int i = 0; i < 3; i++) {
        uc[i] = TO_BYTE(s[i]);
        //uc[i] = (unsigned char)(Clamp(s[i]) * 255);
    }
    if (len == 4) {
        //uc[3] = 255;
        uc[3] = TO_BYTE(1);
    }
#undef TO_BYTE
}


#endif // !__SPECTRUM_H
