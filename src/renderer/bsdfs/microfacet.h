#pragma once

#include "renderer/bsdfs/optical.h"

namespace TrowbridgeReitzDistribution
{
    inline HOST_DEVICE
    float Lambda(
        const float3& w, 
        const float& alpha_x, 
        const float& alpha_y) 
    {
        float absTanTheta = abs(TanTheta(w));
        if (isinf(absTanTheta)) return 0.;
        // Compute _alpha_ for direction _w_
        float alpha =
            sqrt(Cos2Phi(w) * alpha_x * alpha_x + Sin2Phi(w) * alpha_y * alpha_y);
        float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
        return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
    }

    inline HOST_DEVICE
    float G1(
        const float3& w,
        const float& alpha_x,
        const float& alpha_y)
    {
        return 1 / (1 + Lambda(w, alpha_x, alpha_y));
    }

    inline HOST_DEVICE
    float G(
        const float3& wo, 
        const float3& wi, 
        const float& alpha_x,
        const float& alpha_y)
    {
        return 1 / (1 + Lambda(wo, alpha_x, alpha_y) + Lambda(wi, alpha_x, alpha_y));
    }

    inline HOST_DEVICE
    void TrowbridgeReitzSample11(
        float cosTheta, 
        float U1, 
        float U2,
        float* slope_x, 
        float* slope_y)
    {
        // special case (normal incidence)
        if (cosTheta > .9999) {
            float r = sqrt(U1 / (1 - U1));
            float phi = 6.28318530718 * U2;
            *slope_x = r * cos(phi);
            *slope_y = r * sin(phi);
            return;
        }

        float sinTheta =
            sqrt(max((float)0, (float)1 - cosTheta * cosTheta));
        float tanTheta = sinTheta / cosTheta;
        float a = 1 / tanTheta;
        float G1 = 2 / (1 + sqrt(1.f + 1.f / (a * a)));

        // sample slope_x
        float A = 2 * U1 / G1 - 1;
        float tmp = 1.f / (A * A - 1.f);
        if (tmp > 1e10) tmp = 1e10;
        float B = tanTheta;
        float D = sqrt(max(float(B * B * tmp * tmp - (A * A - B * B) * tmp), float(0)));
        float slope_x_1 = B * tmp - D;
        float slope_x_2 = B * tmp + D;
        *slope_x = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

        // sample slope_y
        float S;
        if (U2 > 0.5f) {
            S = 1.f;
            U2 = 2.f * (U2 - .5f);
        }
        else {
            S = -1.f;
            U2 = 2.f * (.5f - U2);
        }
        float z =
            (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
            (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
        *slope_y = S * z * sqrt(1.f + *slope_x * *slope_x);
    }

    inline HOST_DEVICE
    float3 TrowbridgeReitzSample(
        const float3& wi, 
        float alpha_x,
        float alpha_y, 
        float U1, 
        float U2)
    {
        // 1. stretch wi
        float3 wiStretched =
            normalize(make_float3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

        // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
        float slope_x, slope_y;
        TrowbridgeReitzSample11(CosTheta(wiStretched), U1, U2, &slope_x, &slope_y);

        // 3. rotate
        float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
        slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
        slope_x = tmp;

        // 4. unstretch
        slope_x = alpha_x * slope_x;
        slope_y = alpha_y * slope_y;

        // 5. compute normal
        return normalize(make_float3(-slope_x, -slope_y, 1.));
    }

    inline HOST_DEVICE
    float3 Sample_wh(
        const float3& wo, 
        const float2& u,
        const float& alpha_x,
        const float& alpha_y) 
    {
        bool flip = wo.z < 0;
        float3 wh = TrowbridgeReitzSample(flip ? -wo : wo, alpha_x, alpha_y, u.x, u.y);
        if (flip) wh = -wh;
        return wh;
    }

    inline HOST_DEVICE
    float D(
        const float3& wh,
        const float& alpha_x,
        const float& alpha_y)
    {
        float tan2Theta = Tan2Theta(wh);
        if (isinf(tan2Theta)) return 0.;
        const float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
        float e =
            (Cos2Phi(wh) / (alpha_x * alpha_x) + Sin2Phi(wh) / (alpha_y * alpha_y)) *
            tan2Theta;
        return 1 / (PI * alpha_x * alpha_y * cos4Theta * (1 + e) * (1 + e));
    }

    inline HOST_DEVICE
    float Pdf(
        const float3& wo, 
        const float3& wh,
        const float& alpha_x,
        const float& alpha_y)
    {
        return D(wh, alpha_x, alpha_y) * AbsCosTheta(wh);
    }
}


inline HOST_DEVICE
void MicrofacetReflectEval(
    const float3& wo,
    const float3& wi,
    Spectrum* f,
    const Spectrum& color,
    const float& alpha_x,
    const float& alpha_y,
    const Fresnel& fresnel)
{
    using namespace TrowbridgeReitzDistribution;

    *f = 0;

    float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
    float3 wh = wi + wo;
    // Handle degenerate cases for microfacet reflection
    if (cosThetaI == 0 || cosThetaO == 0) return;
    if (wh.x == 0 && wh.y == 0 && wh.z == 0) return;
    wh = normalize(wh);
    Spectrum F = fresnel.eval(dot(wi, wh));
    *f = color * D(wh, alpha_x, alpha_y) * G(wo, wi, alpha_x, alpha_y) * F / 
        (4 * cosThetaO);
}

inline HOST_DEVICE
void MicrofacetReflectPdf(
    const float3& wo,
    const float3& wi,
    float* pdf,
    const Spectrum& color,
    const float& alpha_x,
    const float& alpha_y)
{
    using namespace TrowbridgeReitzDistribution;

    *pdf = 0;

    if (!SameHemisphere(wo, wi)) return;
    float3 wh = normalize(wo + wi);
    *pdf = D(wh, alpha_x, alpha_y)* G1(wi, alpha_x, alpha_y) / (4.0f * wi.z);
    *pdf = Pdf(wo, wh, alpha_x, alpha_y) / (4 * dot(wo, wh));
}

inline HOST_DEVICE
void MicrofacetReflectSample(
    const float2& u,
    const float3& wo,
    float3* wi,
    Spectrum* f,
    float* pdf,
    const Spectrum& color,
    const float& alpha_x,
    const float& alpha_y,
    const Fresnel& fresnel)
{
    using namespace TrowbridgeReitzDistribution;

    *f = *pdf = 0;

    if (wo.z == 0) return;
    float3 wh = Sample_wh(wo, u, alpha_x, alpha_y);
    *wi = Reflect(wo, wh);
    if (!SameHemisphere(wo, *wi)) return;
    *pdf = D(wh, alpha_x, alpha_y) * G1(*wi, alpha_x, alpha_y) / (4.0f * wi->z);
    *pdf = Pdf(wo, wh, alpha_x, alpha_y) / (4 * dot(wo, wh));
    MicrofacetReflectEval(wo, *wi, f, color, alpha_x, alpha_y, fresnel);
}