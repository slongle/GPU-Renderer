#include "framebuffer.h"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb_image/stb_image.h"
#endif // !STB_IMAGE_IMPLEMENTATION

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ext/stb_image/stb_image_write.h"
#endif // !STB_IMAGE_WRITE_IMPLEMENTATION

FrameBufferView::FrameBufferView(const FrameBuffer* fb):
    m_resolution_x(fb->m_resolution_x),
    m_resolution_y(fb->m_resolution_y),
    m_buffer(fb->m_buffer.data()),
    m_sample_num(fb->m_sample_num.data())
    {}

void FrameBuffer::output(const std::string& filename)
{
    Buffer<HOST_BUFFER, Spectrum> buffer(m_buffer);
    Buffer<HOST_BUFFER, uint32> sample_num(m_sample_num);

    Buffer<HOST_BUFFER, uint8> out(buffer.size() * 3);
    for (int i = 0; i < buffer.size(); i++) {
        Spectrum color = buffer[i] / sample_num[i];
        uint8* ptr_uc = out.data() + i * 3;
        ptr_uc[0] = (uint8)clamp(255.f * GammaCorrect(color.x) + 0.5f, 0.f, 255.f);
        ptr_uc[1] = (uint8)clamp(255.f * GammaCorrect(color.y) + 0.5f, 0.f, 255.f);
        ptr_uc[2] = (uint8)clamp(255.f * GammaCorrect(color.z) + 0.5f, 0.f, 255.f);
    }

    stbi_flip_vertically_on_write(true);
    stbi_write_png(filename.c_str(), m_resolution_x, m_resolution_y, 3, out.data(), 0);
}
