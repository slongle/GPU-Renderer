#pragma once

#include<filesystem>

#include "renderer/fwd.h"
#include "renderer/triangle.h"
#include "renderer/material.h"
#include "renderer/loaders/propertylist.h"

#include "ext/pugixml/pugixml.hpp"

void load_mitsuba_file(const std::string& filename, std::vector<Triangle>& triangles);

enum ETag {
    EHide,
    EScene,
    EMode,

    EIntegrator,
    ECamera,
    ESampler,
    EFilm,
    ERFilter,
    ELight,
    EShape,
    EBSDF,
    EVolume,
    EMedium,
    ERef,

    EBoolean,
    EInteger,
    EFloat,
    EString,
    EVector,
    EColor,
    ETransform,
    ELookAt,
    ETranslate,
    EScale,
    ERotate,
    EMatrix
};

class ParseRecord {
public:
    ParseRecord(const std::string filename, std::vector<Triangle>* triangles);

public:
    const std::string m_filename;
    Transform m_transform;
    std::map<std::string, ETag> m_tags;
    std::filesystem::path m_path;
    std::vector<Triangle>* m_triangles;

public:
    Material m_current_material;
    std::map<std::string, Material> m_named_material;
    Spectrum m_current_light;

    void createMaterial(const std::string& type, const PropertyList& list);
    void createNamedMaterial(const std::string& id, const std::string& type, const PropertyList& list);
    void createLight(const std::string& type, const PropertyList& list);
    void createShape(const std::string& type, const PropertyList& list);
};

bool HasAttribute(const pugi::xml_node& node, const std::string& name);
std::string GetOffset(ptrdiff_t pos, ParseRecord& record);
void ParseTag(pugi::xml_node& node, PropertyList& fatherList, ParseRecord& record);
void HandleTag(pugi::xml_node& node, PropertyList& myList, PropertyList& fatherList, ParseRecord& record);
void Parse(ParseRecord& record);