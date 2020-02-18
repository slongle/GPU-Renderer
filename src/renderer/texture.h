#pragma once

#include "renderer/fwd.h"
#include "renderer/spectrum.h"

enum TextureType
{
    TEXTURE_SPECTRUM,
    TEXTURE_FLOAT,
};

class Texture;

class TextureView
{
public:
    float2 m_uvoffset, m_uvscale;
    uint32 m_width, m_height;
    Spectrum* m_buffer;
    TextureType m_type;

    TextureView();
    TextureView(const Texture* texture);

    HOST_DEVICE
    Spectrum evalSpectrum(const float2& uv) const
    {
        float2 now_uv = uv * m_uvscale + m_uvoffset;
        int u2 = int(now_uv.x * m_width) % m_width, v2 = int(now_uv.y * m_height) % m_height;
        int idx = v2 * m_width + u2;
        return m_buffer[idx];
    }

    HOST_DEVICE
    float evalFloat(const float2& uv) const
    {       
        float2 now_uv = uv * m_uvscale + m_uvoffset;
        int u2 = int(now_uv.x * m_width) % m_width, v2 = int(now_uv.y * m_height) % m_height;
        int idx = v2 * m_width + u2;
        return m_buffer[idx].r;
    }
};

class Texture
{
public:
    // Single value texture
    Texture(const float& f);
    Texture(const Spectrum& s);
    // Bitmap texture
    Texture(
        const std::string& filename, 
        const float2& uvoffset = make_float2(0.f), 
        const float2& uvscale = make_float2(1.f));
    // Checkerboard texture
    Texture(
        const Spectrum& c0, const Spectrum& c1,
        const float2& uvoffset = make_float2(0.f),
        const float2& uvscale = make_float2(1.f));

    TextureView view() const { return TextureView(this); }

public:
    std::vector<Spectrum> m_cpu_buffer;
    Buffer<DEVICE_BUFFER, Spectrum> m_gpu_buffer;

    float2 m_uvoffset, m_uvscale;
    int m_width, m_height;

    TextureType m_type;
};