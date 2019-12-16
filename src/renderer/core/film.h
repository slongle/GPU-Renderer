#pragma once
#ifndef __FILM_H
#define __FILM_H

#include "renderer/core/fwd.h"
#include "renderer/core/transform.h"
#include "renderer/core/parameterset.h"
#include "renderer/core/spectrum.h"

class Film {
public:
    Film() {}
    Film(Point2i resolution, std::string filename);

    void SetVal(int x, int y, Spectrum v);
    void Output() const;

    std::string m_filename; 
    Point2i m_resolution;
    int m_channels;
    unsigned char* m_bitmap = nullptr;
};

inline
std::shared_ptr<Film>
CreateFilm(
    const ParameterSet& param);

inline
Film::Film(
    Point2i resolution,
    std::string filename)
    : m_resolution(resolution), m_filename(filename), m_channels(4)
{
    m_bitmap = new unsigned char[m_resolution.x * m_resolution.y * 4];
}

inline void Film::SetVal(int x, int y, Spectrum v)
{
    int index = y * m_resolution.x + x;
    SpectrumToUnsignedChar(v, &m_bitmap[index * m_channels], m_channels);
}

inline
std::shared_ptr<Film>
CreateFilm(
    const ParameterSet& param)
{
    Point2i resolution(param.GetInt("xresolution"), param.GetInt("yresolution"));
    std::string filename(param.GetString("filename"));
    return std::shared_ptr<Film>(new Film(resolution, filename));
}



#endif // !__FILM_H
