#pragma once

#include "renderer/fwd.h"
#include "renderer/spectrum.h"

template<typename T>
inline __device__ __host__
void Swap(T& a, T& b) {
    T c(a);
    a = b;
    b = c;
}

inline __device__ __host__
float Clamp(float a, float l, float r) {
    if (a > r) return r;
    else if (a < l) return l;
    else return a;
}

inline HOST_DEVICE
float FrDielectric(float cosThetaI, float etaI, float etaT) {
    cosThetaI = Clamp(cosThetaI, -1.f, 1.f);
    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;

    if (!entering) {
        Swap(etaI, etaT);
        cosThetaI = abs(cosThetaI);
    }

    // Compute _cosThetaT_ using Snell's law
    float sinThetaI = sqrt(fmaxf(0.f, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    // Handle total internal reflection
    if (sinThetaT >= 1) return 1.f;
    float cosThetaT = sqrt(fmaxf(0.f, 1 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) * 0.5f;
}

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
inline HOST_DEVICE
Spectrum FrConductor(
    float cosThetaI,
    const Spectrum& etai,
    const Spectrum& etat,
    const Spectrum& k)
{
    cosThetaI = Clamp(cosThetaI, -1.f, 1.f);
    Spectrum eta = etat / etai;
    Spectrum etak = k / etai;

    float cosThetaI2 = cosThetaI * cosThetaI;
    float sinThetaI2 = 1.f - cosThetaI2;
    Spectrum eta2 = eta * eta;
    Spectrum etak2 = etak * etak;

    Spectrum t0 = eta2 - etak2 - sinThetaI2;
    Spectrum a2plusb2 = sqrt(t0 * t0 + eta2 * etak2 * 4);
    Spectrum t1 = a2plusb2 + cosThetaI2;
    Spectrum a = sqrt((a2plusb2 + t0) * 0.5);
    Spectrum t2 = a * (float)2 * cosThetaI;
    Spectrum Rs = (t1 - t2) / (t1 + t2);

    Spectrum t3 = a2plusb2 * cosThetaI2 + sinThetaI2 * sinThetaI2;
    Spectrum t4 = t2 * sinThetaI2;
    Spectrum Rp = Rs * (t3 - t4) / (t3 + t4);

    return (Rp + Rs) * 0.5f;
}