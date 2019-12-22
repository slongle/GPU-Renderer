#pragma once
#ifndef __OPTIC_H
#define __OPTIC_H

#include "renderer/core/geometry.h"
#include "renderer/core/spectrum.h"

inline __host__ __device__
Vector3f Reflect(const Vector3f& wo, const Vector3f& n) {
    return -wo + n * 2 * Dot(wo, n);
}

inline __host__ __device__
bool Refract(
    const Vector3f& wi, 
    const Normal3f& n, 
    Float eta,
    Vector3f* wt) 
{
    // Compute $\cos \theta_\roman{t}$ using Snell's law
    Float cosThetaI = Dot(n, wi);
    Float sin2ThetaI = max(Float(0), Float(1 - cosThetaI * cosThetaI));
    Float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return false;
    Float cosThetaT = sqrt(1 - sin2ThetaT);
    *wt = -wi * eta +  Vector3f(n) * (eta * cosThetaI - cosThetaT);
    return true;
}

inline __host__ __device__
Float FrDielectric(Float cosThetaI, Float etaI, Float etaT) {
    cosThetaI = Clamp(cosThetaI, -1, 1);
    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;
    
    if (!entering) {
        Swap(etaI, etaT);
        cosThetaI = abs(cosThetaI);
    }

    // Compute _cosThetaT_ using Snell's law
    Float sinThetaI = sqrt(max((Float)0, 1 - cosThetaI * cosThetaI));
    Float sinThetaT = etaI / etaT * sinThetaI;

    // Handle total internal reflection
    if (sinThetaT >= 1) return 1;
    Float cosThetaT = sqrt(max((Float)0, 1 - sinThetaT * sinThetaT));
    Float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    Float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) * 0.5f;
}

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
inline __host__ __device__
Spectrum FrConductor(
    Float cosThetaI, 
    const Spectrum& etai,
    const Spectrum& etat, 
    const Spectrum& k) 
{
    cosThetaI = Clamp(cosThetaI, -1, 1);
    Spectrum eta = etat / etai;
    Spectrum etak = k / etai;

    Float cosThetaI2 = cosThetaI * cosThetaI;
    Float sinThetaI2 = 1. - cosThetaI2;
    Spectrum eta2 = eta * eta;
    Spectrum etak2 = etak * etak;

    Spectrum t0 = eta2 - etak2 - sinThetaI2;
    Spectrum a2plusb2 = Sqrt(t0 * t0 + eta2 * etak2 * 4);
    Spectrum t1 = a2plusb2 + cosThetaI2;
    Spectrum a = Sqrt((a2plusb2 + t0) * 0.5);
    Spectrum t2 = a * (Float)2 * cosThetaI;
    Spectrum Rs = (t1 - t2) / (t1 + t2);

    Spectrum t3 = a2plusb2 * cosThetaI2 + sinThetaI2 * sinThetaI2;
    Spectrum t4 = t2 * sinThetaI2;
    Spectrum Rp = Rs * (t3 - t4) / (t3 + t4);

    return (Rp + Rs) * 0.5f;
}


#endif // !__OPTIC_H
