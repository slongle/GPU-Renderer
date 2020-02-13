#pragma once

#include "renderer/fwd.h"

class RGBSpectrum
{
public:
    HOST_DEVICE
    RGBSpectrum(float c = 0) : r(c), g(c), b(c) {}
    HOST_DEVICE
    RGBSpectrum(float r, float g, float b) :r(r), g(g), b(b) {}
    HOST_DEVICE
    RGBSpectrum(float3 s) : r(fabsf(s.x)), g(fabsf(s.y)), b(fabsf(s.z)) {}

    HOST_DEVICE
    RGBSpectrum operator - () const { return RGBSpectrum(-r, -g, -b); }
    HOST_DEVICE
    RGBSpectrum operator + (const RGBSpectrum& s) const { return RGBSpectrum(r + s.r, g + s.g, b + s.b); }
    HOST_DEVICE
    RGBSpectrum& operator += (const RGBSpectrum& s) { r += s.r, g += s.g, b += s.b; return *this; }
    HOST_DEVICE
    RGBSpectrum operator - (const RGBSpectrum& s) const { return RGBSpectrum(r - s.r, g - s.g, b - s.b); }
    HOST_DEVICE
    RGBSpectrum& operator -= (const RGBSpectrum& s) { r -= s.r, g -= s.g, b -= s.b; return *this; }
    HOST_DEVICE
    RGBSpectrum operator * (const float& s) const { return RGBSpectrum(r * s, g * s, b * s); }
    HOST_DEVICE
    RGBSpectrum operator * (const RGBSpectrum& s) const { return RGBSpectrum(r * s.r, g * s.g, b * s.b); }
    HOST_DEVICE
    RGBSpectrum& operator *= (const float& s) { r *= s, g *= s, b *= s; return *this; }
    HOST_DEVICE
    RGBSpectrum& operator *= (const RGBSpectrum& s) { r *= s.r, g *= s.g, b *= s.b; return *this; }
    HOST_DEVICE
    RGBSpectrum operator / (const float& s) const { return RGBSpectrum(r / s, g / s, b / s); }
    HOST_DEVICE
    RGBSpectrum operator / (const RGBSpectrum& s) const { return RGBSpectrum(r / s.r, g / s.g, b / s.b); }
    HOST_DEVICE
    RGBSpectrum& operator /= (const float& s) { r /= s, g /= s, b /= s; return *this; }
    HOST_DEVICE
    RGBSpectrum& operator /= (const RGBSpectrum& s) { r /= s.r, g /= s.g, b /= s.b; return *this; }
    float r, g, b;
};

inline HOST_DEVICE
RGBSpectrum sqrt(const RGBSpectrum& s) { return RGBSpectrum(sqrt(s.r), sqrt(s.g), sqrt(s.b)); }
inline HOST_DEVICE
float fmaxf(const RGBSpectrum& s) { return max(s.r, max(s.g, s.b)); }
inline HOST_DEVICE
RGBSpectrum fabs(const RGBSpectrum& s) { return RGBSpectrum(fabsf(s.r), fabsf(s.g), fabsf(s.b)); }

typedef RGBSpectrum Spectrum;

inline HOST_DEVICE
bool isBlack(const Spectrum& s) 
{
    return s.r == 0 && s.g == 0 && s.b == 0;
}

inline HOST_DEVICE
float GammaCorrect(float value) 
{
    if (value <= 0.0031308f)
        return 12.92f * value;
    /*if (isnan(powf(value, (1.f / 2.4f)))) {
        printf("%f\n", value);

    }*/
    return 1.055f * powf(value, (1.f / 2.4f)) - 0.055f;
}
