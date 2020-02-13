#pragma once

#include "renderer/fwd.h"
#include "renderer/sampling.h"
#include "renderer/spectrum.h"

class LambertReflect
{
public:
    HOST_DEVICE
    LambertReflect() {}
    HOST_DEVICE
    LambertReflect(Spectrum color):m_color(color) {}

    /// evaluate f = BSDF * cos, BSDF = color / PI
    ///
    HOST_DEVICE
    Spectrum eval(
        const float3    wo,
        const float3    wi) const
    {
        return m_color * INV_PI * fabsf(wi.z);
    }

    /// evaluate pdf
    ///
    HOST_DEVICE
    void pdf(
        const float3    wo,
        const float3    wi,
              float*    pdf) const
    {
        *pdf = fabsf(wi.z) * INV_PI;
    }

    /// sample wi, evaluate f and pdf
    ///
    HOST_DEVICE
    void sample(
        const float2    s,
        const float3    wo,
              float3*   wi,
              Spectrum* f,
              float*    pdf) const
    {
        *wi = CosineSampleHemisphere(s);
        *f = m_color * INV_PI * fabsf(wi->z);
        *pdf = fabsf(wi->z) * INV_PI;
    }

//private:
    Spectrum m_color;
};