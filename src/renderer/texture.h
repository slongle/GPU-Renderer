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
    uint32 m_width, m_height;
    Spectrum* m_buffer;
    TextureType m_type;

    TextureView();
    TextureView(const Texture* texture);

    HOST_DEVICE
    Spectrum evalSpectrum(const float2& uv) const
    {
        int u2 = int(uv.x * m_width) % m_width, v2 = int(uv.y * m_height) % m_height;
        int idx = v2 * m_width + u2;
        return m_buffer[idx];
    }

    HOST_DEVICE
    float evalFloat(const float2& uv) const
    {       
        int u2 = int(uv.x * m_width) % m_width, v2 = int(uv.y * m_height) % m_height;
        int idx = v2 * m_width + u2;
        return m_buffer[idx].r;
    }
};

class Texture
{
public:
    Texture(const float& f);
    Texture(const Spectrum& s);
    Texture(const std::string& filename);

    TextureView view() const { return TextureView(this); }

public:
    std::vector<Spectrum> m_cpu_buffer;
    Buffer<DEVICE_BUFFER, Spectrum> m_gpu_buffer;

    int m_width, m_height;

    TextureType m_type;
};