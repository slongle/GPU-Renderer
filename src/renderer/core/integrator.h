#pragma once
#ifndef __INTEGRATOR_H
#define __INTEGRATOR_H

#include "renderer/core/fwd.h"
#include "renderer/core/parameterset.h"

class Integrator {
public:
    Integrator() {}
    Integrator(int maxDepth);

    int m_maxDepth;
};

std::shared_ptr<Integrator>
CreateIntegrator(
    const ParameterSet& param);

#endif // !__INTEGRATOR_H
