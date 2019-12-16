#include "film.h"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb_image/stb_image.h"
#endif // !STB_IMAGE_IMPLEMENTATION

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ext/stb_image/stb_image_write.h"
#endif // !STB_IMAGE_WRITE_IMPLEMENTATION

void Film::Output() const
{
    stbi_write_png(m_filename.c_str(), m_resolution.x, m_resolution.y, m_channels, m_bitmap, 0);
}