#pragma once
#ifndef __MICROFACET_H
#define __MICROFACET_H

#include "renderer/core/sampling.h"

class GGXDistribution {
public:
    GGXDistribution() {}
    GGXDistribution(const Float& uroughness, const Float& vroughness);

    bool isSpecular() const;

    Float D(
        const Vector3f& wh) const;
    Vector3f Sample_wh(
        const Vector3f& wo, 
        unsigned int& seed) const;
    Float Pdf(
        const Vector3f& wo,
        const Vector3f& wh) const;
    Float G1(const Vector3f& w) const;
    Float Lambda(const Vector3f& w) const;
    Float G(const Vector3f& wo, const Vector3f& wi) const;

    static void GGXSample11(
        Float cosTheta,
        Float U1,
        Float U2,
        Float* slope_x,
        Float* slope_y);

    static Vector3f GGXSample(
        const Vector3f& wi, 
        Float alpha_x,
        Float alpha_y, 
        Float U1, 
        Float U2);

    Float m_alphax, m_alphay;
};

inline
GGXDistribution::GGXDistribution(
    const Float& uroughness, 
    const Float& vroughness)
    : m_alphax(uroughness), m_alphay(vroughness)
{
}

inline __host__ __device__
bool GGXDistribution::isSpecular() const
{
    return m_alphax == 0 && m_alphay == 0;
}

inline __host__ __device__
Float GGXDistribution::D(const Vector3f& wh) const
{
    Float tan2Theta = Tan2Theta(wh);
    if (isinf(tan2Theta)) return 0.f;
    const Float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
    Float e =
        (Cos2Phi(wh) / (m_alphax * m_alphax) + Sin2Phi(wh) / (m_alphay * m_alphay)) *
        tan2Theta;
    return 1 / (Pi * m_alphax * m_alphay * cos4Theta * (1 + e) * (1 + e));
}

inline __host__ __device__
Vector3f GGXDistribution::Sample_wh(const Vector3f& wo, unsigned int& seed) const
{
    Vector3f wh;
    bool flip = wo.z < 0;
    Vector3f wi;
    if (flip) wi = -wo;
    else wi = wo;
    wh = GGXSample(wi, m_alphax, m_alphay, NextRandom(seed), NextRandom(seed));    
    if (flip) wh = -wh;
    return wh;
}

inline __host__ __device__
Float GGXDistribution::Pdf(
    const Vector3f& wo, 
    const Vector3f& wh) const
{
    return D(wh) * G1(wo) * AbsDot(wo, wh) / AbsCosTheta(wo);
}

inline __host__ __device__
Float GGXDistribution::G1(const Vector3f& w) const
{
    return 1 / (1 + Lambda(w));
}

inline __host__ __device__
Float GGXDistribution::Lambda(const Vector3f& w) const
{
    Float absTanTheta = abs(TanTheta(w));
    if (isinf(absTanTheta)) return 0.f;
    // Compute _alpha_ for direction _w_
    Float alpha =
        sqrt(Cos2Phi(w) * m_alphax * m_alphax + Sin2Phi(w) * m_alphay * m_alphay);
    Float a = 1 / (alpha * absTanTheta);
    if (a >= 1.6f) return 0.f;
    return (1 - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
}

inline __host__ __device__
Float GGXDistribution::G(
    const Vector3f& wo, 
    const Vector3f& wi) const
{
    return 1 / (1 + Lambda(wo) + Lambda(wi));
}

inline __host__ __device__
void GGXDistribution::GGXSample11(
    Float cosTheta, 
    Float U1, 
    Float U2, 
    Float* slope_x, 
    Float* slope_y)
{
    // special case (normal incidence)
    if (cosTheta > .9999f) {
        Float r = sqrt(U1 / (1 - U1));
        Float phi = 6.28318530718f * U2;
        *slope_x = r * cos(phi);
        *slope_y = r * sin(phi);
        return;
    }

    Float sinTheta =
        sqrt(max((Float)0, (Float)1 - cosTheta * cosTheta));
    Float tanTheta = sinTheta / cosTheta;
    Float a = 1 / tanTheta;
    Float G1 = 2 / (1 + sqrt(1.f + 1.f / (a * a)));

    // sample slope_x
    Float A = 2 * U1 / G1 - 1;
    Float tmp = 1.f / (A * A - 1.f);
    if (tmp > 1e10) tmp = 1e10f;
    Float B = tanTheta;
    Float D = sqrt(
        max(Float(B * B * tmp * tmp - (A * A - B * B) * tmp), Float(0)));
    Float slope_x_1 = B * tmp - D;
    Float slope_x_2 = B * tmp + D;
    *slope_x = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

    // sample slope_y
    Float S;
    if (U2 > 0.5f) {
        S = 1.f;
        U2 = 2.f * (U2 - .5f);
    }
    else {
        S = -1.f;
        U2 = 2.f * (.5f - U2);
    }
    Float z =
        (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
        (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
    *slope_y = S * z * sqrt(1.f + *slope_x * *slope_x);
}

inline __host__ __device__
Vector3f GGXDistribution::GGXSample(
    const Vector3f& wi, 
    Float alpha_x, 
    Float alpha_y, 
    Float U1, 
    Float U2)
{
    // 1. stretch wi
    Vector3f wiStretched =
        Normalize(Vector3f(alpha_x * wi.x, alpha_y * wi.y, wi.z));

    // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
    Float slope_x, slope_y;
    GGXSample11(CosTheta(wiStretched), U1, U2, &slope_x, &slope_y);

    // 3. rotate
    Float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
    slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
    slope_x = tmp;

    // 4. unstretch
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;

    // 5. compute normal
    return Normalize(Vector3f(-slope_x, -slope_y, 1.));
}

#endif // !__MICROFACET_H