#include "integrator.h"

Integrator::Integrator(
    int maxDepth)
    : m_maxDepth(maxDepth)
{
}

std::shared_ptr<Integrator> 
CreateIntegrator(
    const ParameterSet& param)
{
    int maxDepth = param.GetInt("maxdepth", 5);
    return std::make_shared<Integrator>(maxDepth);
}

