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

    __host__ __device__ void SetVal(int x, int y, Spectrum v);
    void Output() const;

    std::string m_filename; 
    Point2i m_resolution;
    int m_channels;
    unsigned char* m_bitmap = nullptr;
};

std::shared_ptr<Film>
CreateFilm(
    const ParameterSet& param);

inline __host__ __device__
void Film::SetVal(int x, int y, Spectrum v)
{
    int index = y * m_resolution.x + x;
    SpectrumToUnsignedChar(v, &m_bitmap[index * m_channels], m_channels);
}





#endif // !__FILM_H
