#pragma once

#include "renderer/spectrum.h"
#include "renderer/sampling.h"

inline HOST_DEVICE
float3 Reflect(float3 wo, float3 n) {
    return -wo + n * 2 * dot(wo, n);
}

inline HOST_DEVICE
bool Refract(
    float3 wi,
    float3 n,
    float eta,
    float3* wt)
{
    // Compute $\cos \theta_\roman{t}$ using Snell's law
    float cosThetaI = dot(n, wi);
    float sin2ThetaI = max(0.f, 1 - cosThetaI * cosThetaI);
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return false;
    float cosThetaT = sqrt(1 - sin2ThetaT);
    *wt = -wi * eta + float3(n) * (eta * cosThetaI - cosThetaT);
    return true;
}

inline HOST_DEVICE
float FrDielectric(float cosThetaI, float etaI, float etaT) {
    cosThetaI = clamp(cosThetaI, -1.f, 1.f);
    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;

    if (!entering) {
        // swap etaT and etaI
        float tmp = etaT;
        etaT = etaI;
        etaI = tmp;

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
    cosThetaI = clamp(cosThetaI, -1.f, 1.f);
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