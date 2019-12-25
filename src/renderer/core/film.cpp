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
    : m_resolution(resolution), m_filename(filename), m_channels(3)
{    
    m_bitmap = new  Float[m_resolution.x * m_resolution.y * 3];
    m_sampleNum = new unsigned int[m_resolution.x * m_resolution.y];
    memset(m_bitmap, 0, sizeof(Float) * m_resolution.x * m_resolution.y * 3);
    memset(m_sampleNum, 0, sizeof(unsigned int) * m_resolution.x * m_resolution.y);    
}

void Film::Output()
{
    printf("Output\n");
    ExportToUnsignedChar();
    stbi_write_png(m_filename.c_str(), m_resolution.x, m_resolution.y, m_channels, m_bitmapOutput, 0);
    printf("Done\n");
}

void Film::DrawLine(
    const Point2f& s, 
    const Point2f& t,
    const Spectrum& col)
{
    int x0 = s.x, y0 = s.y;
    int x1 = t.x, y1 = t.y;
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = (dx > dy ? dx : -dy) / 2;

    while (SetVal(x0, y0, col), x0 != x1 || y0 != y1) {
        int e2 = err;
        if (e2 > -dx) { err -= dy; x0 += sx; }
        if (e2 < dy) { err += dx; y0 += sy; }
    }
}

void Film::ExportToUnsignedChar() 
{
    m_bitmapOutput = new unsigned char[m_resolution.x * m_resolution.y * m_channels];
    for (int x = 0; x < m_resolution.x; x++) {
        for (int y = 0; y < m_resolution.y; y++) {
            int index = y * m_resolution.x + x;
            Spectrum v(m_bitmap[index * 3], m_bitmap[index * 3 + 1], m_bitmap[index * 3 + 2]);
            if (m_sampleNum[index] != 0) {
                v /= m_sampleNum[index];
            }
            SpectrumToUnsignedChar(v, &m_bitmapOutput[index * m_channels], m_channels);
        }
    }
}

std::shared_ptr<Film>
CreateFilm(
    const ParameterSet& param)
{
    Point2i resolution(param.GetInt("xresolution"), param.GetInt("yresolution"));
    std::string filename(param.GetString("filename"));
    return std::shared_ptr<Film>(new Film(resolution, filename));
}
