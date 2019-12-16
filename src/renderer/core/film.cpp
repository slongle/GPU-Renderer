#include "film.h"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb_image/stb_image.h"
#endif // !STB_IMAGE_IMPLEMENTATION

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ext/stb_image/stb_image_write.h"
#endif // !STB_IMAGE_WRITE_IMPLEMENTATION

Film::Film(
    Point2i resolution,
    std::string filename)
    : m_resolution(resolution), m_filename(filename), m_channels(4)
{
    m_bitmap = new unsigned char[m_resolution.x * m_resolution.y * 4];
}

void Film::Output() const
{
    stbi_write_png(m_filename.c_str(), m_resolution.x, m_resolution.y, m_channels, m_bitmap, 0);
}

std::shared_ptr<Film>
CreateFilm(
    const ParameterSet& param)
{
    Point2i resolution(param.GetInt("xresolution"), param.GetInt("yresolution"));
    std::string filename(param.GetString("filename"));
    return std::shared_ptr<Film>(new Film(resolution, filename));
}
