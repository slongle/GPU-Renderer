#include "film.h"

Film::Film(
    Point2i resolution,
    std::string filename)
    : m_resolution(resolution), m_filename(filename)
{
    //m_bitmap = new unsigned char[m_resolution.x * m_resolution.y];
}

std::shared_ptr<Film> 
CreateFilm(
    const ParameterSet& param)
{
    Point2i resolution(param.GetInt("xresolution"), param.GetInt("yresolution"));
    std::string filename(param.GetString("filename"));
    return std::shared_ptr<Film>(new Film(resolution, filename));
}

