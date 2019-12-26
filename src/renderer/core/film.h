#pragma once
#ifndef __FILM_H
#define __FILM_H

#include "renderer/core/transform.h"
#include "renderer/core/parameterset.h"
#include "renderer/core/spectrum.h"

class Film {
public:
    Film() {}
    Film(Point2i resolution, std::string filename);

    __host__ __device__ void SetVal(int x, int y, Spectrum v);
    __host__ __device__ void AddSample(int x, int y, Spectrum v);
    __host__ __device__ Spectrum GetPixelSpectrum(int x, int y)const;
    __host__ __device__ Spectrum GetPixelSpectrum(int index)const;
    void DrawLine(const Point2f& s, const Point2f& t, const Spectrum& col);
    void ExportToUnsignedChar();
    void Output();

    std::string m_filename; 
    Point2i m_resolution;
    int m_channels;
    Float* m_bitmap = nullptr;
    unsigned char* m_bitmapOutput = nullptr;
    unsigned int* m_sampleNum = nullptr;
};

std::shared_ptr<Film>
CreateFilm(
    const ParameterSet& param);

inline __host__ __device__
void Film::SetVal(int x, int y, Spectrum v)
{
    int index = y * m_resolution.x + x;    
    for (int i = 0; i < 3; i++) {
        m_bitmap[index * 3 + i] = v[i];
    }    
    m_sampleNum[index] = 1;
}

inline __host__ __device__ 
void Film::AddSample(int x, int y, Spectrum v)
{
    int index = y * m_resolution.x + x;
    for (int i = 0; i < 3; i++) {        
        m_bitmap[index * 3 + i] += v[i];
    }
    m_sampleNum[index]++;
}

inline __host__ __device__
Spectrum Film::GetPixelSpectrum(int index) const
{
    Spectrum v(m_bitmap[index * 3], m_bitmap[index * 3 + 1], m_bitmap[index * 3 + 2]);
    v /= Float(m_sampleNum[index]);
    return v;
}

inline __host__ __device__ 
Spectrum Film::GetPixelSpectrum(int x, int y) const
{
    int index = y * m_resolution.x + x;
    return GetPixelSpectrum(index);   
}

#endif // !__FILM_H
