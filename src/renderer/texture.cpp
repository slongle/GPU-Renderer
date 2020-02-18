#include "texture.h"

TextureView::TextureView()
    : m_uvoffset(make_float2(0.f)), m_uvscale(make_float2(1.f)),
      m_width(0), m_height(0), m_buffer(nullptr)
{
}

TextureView::TextureView(const Texture* texture)
    : m_uvoffset(texture->m_uvoffset), m_uvscale(texture->m_uvscale),
      m_width(texture->m_width), m_height(texture->m_height),
      m_buffer(texture->m_gpu_buffer.data()), m_type(texture->m_type)
{
}

Texture::Texture(const float& f)
    : m_uvoffset(make_float2(0.f)), m_uvscale(make_float2(1.f)), m_width(1), m_height(1), m_type(TEXTURE_FLOAT)
{
    m_cpu_buffer.push_back(Spectrum(f));
    m_gpu_buffer.copyFrom(m_cpu_buffer.size(), HOST_BUFFER, m_cpu_buffer.data());
}

Texture::Texture(const Spectrum& s)
    : m_uvoffset(make_float2(0.f)), m_uvscale(make_float2(1.f)), m_width(1), m_height(1), m_type(TEXTURE_SPECTRUM)
{
    m_cpu_buffer.push_back(s);
    m_gpu_buffer.copyFrom(m_cpu_buffer.size(), HOST_BUFFER, m_cpu_buffer.data());
}

Texture::Texture(
    const std::string& filename,
    const float2& uvoffset, 
    const float2& uvscale)
    : m_uvoffset(uvoffset), m_uvscale(uvscale), m_type(TEXTURE_SPECTRUM)
{
    std::vector<float> buffer;    
    ReadImage(filename, &m_width, &m_height, buffer);    
    m_cpu_buffer.resize(m_width * m_height);
    for (int i = 0; i < m_width * m_height * 3; i += 3) {
        m_cpu_buffer[i / 3] = Spectrum(buffer[i + 0], buffer[i + 1], buffer[i + 2]);
    }
    m_gpu_buffer.copyFrom(m_cpu_buffer.size(), HOST_BUFFER, m_cpu_buffer.data());
}

Texture::Texture(
    const Spectrum& c0, 
    const Spectrum& c1,
    const float2& uvoffset, 
    const float2& uvscale)
    : m_uvoffset(uvoffset), m_uvscale(uvscale), m_width(2), m_height(2), m_type(TEXTURE_SPECTRUM)
{
    m_cpu_buffer.resize(4);
    m_cpu_buffer[0] = m_cpu_buffer[3] = c0;
    m_cpu_buffer[1] = m_cpu_buffer[2] = c1;
    m_gpu_buffer.copyFrom(m_cpu_buffer.size(), HOST_BUFFER, m_cpu_buffer.data());
}
