#pragma once
#ifndef __FILM_H
#define __FILM_H

#include "renderer/core/fwd.h"
#include "renderer/core/transform.h"
#include "renderer/core/parameterset.h"

class Film {

};

std::shared_ptr<Film>
CreateFilm(
    const ParameterSet& param);

#endif // !__FILM_H
