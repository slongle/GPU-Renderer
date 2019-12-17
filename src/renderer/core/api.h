#pragma once
#ifndef __API_H
#define __API_H

#include "renderer/core/fwd.h"
#include "renderer/core/parameterset.h"
#include "renderer/core/renderer.h"

void apiAttributeBegin();
void apiAttributeEnd();
void apiWorldBegin();
std::shared_ptr<Renderer> apiWorldEnd();

void apiTransform(const Float m[16]);

void apiIntegrator(const std::string& type, ParameterSet params);
void apiSampler(const std::string& type, ParameterSet params);
void apiFilter(const std::string& type, ParameterSet params);
void apiFilm(const std::string& type, ParameterSet params);
void apiCamera(const std::string& type, ParameterSet params);
void apiNamedMaterial(const std::string& name, ParameterSet params);
void apiMakeNamedMaterial(const std::string& name, ParameterSet params);
void apiShape(const std::string& type, ParameterSet params);
void apiAreaLightSource(const std::string& type, ParameterSet params);












#endif // !__API_H
