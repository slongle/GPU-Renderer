#include "mitsubaloader.h"
#include "renderer/scene.h"
#include "renderer/loaders/objloader.h"

void load_mitsuba_file(
    const std::string& filename, 
    Scene* scene
    //std::vector<Triangle>& triangles,
    //Camera& camera
    )
{
    ParseRecord record(
        filename, 
        scene
        //&triangles, &camera
        );
    Parse(record);
}

bool toBoolean(const std::string& str) {
    assert(str == "true" || str == "false", "Can't convert " + str + " to Boolean type");
    if (str == "true") return true;
    else return false;
}

int toInteger(const std::string& str) {
    char* end_ptr = nullptr;
    int result = strtol(str.c_str(), &end_ptr, 10);
    assert((*end_ptr == '\0'), "Can't convert " + str + " to Integer type");
    return result;
}

float toFloat(const std::string& str) {
    char* end_ptr = nullptr;
    float result = strtof(str.c_str(), &end_ptr);
    assert((*end_ptr == '\0'), "Can't convert " + str + " to Float type");
    return result;
}

float3 toVector(const std::string& str) {
    float3 ret;
    char* endptr;
    ret.x = strtof(str.c_str(), &endptr); endptr++;
    ret.y = strtof(endptr, &endptr); endptr++;
    ret.z = strtof(endptr, &endptr);
    return ret;
}

Spectrum toColor(const std::string& str) {
    Spectrum ret;
    char* endptr;
    ret.x = strtof(str.c_str(), &endptr); endptr++;
    ret.y = strtof(endptr, &endptr); endptr++;
    ret.z = strtof(endptr, &endptr);
    return ret;
}

void create_cube_triangles(const PropertyList& list, std::vector<Triangle>& triangles)
{
    std::vector<float3> p;
    std::vector<float3> n;
    
    p.push_back(make_float3(0.));
    p.push_back(make_float3(1.000000 , 1.000000 , -1.000000));        
    p.push_back(make_float3(1.000000 , -1.000000, -1.000000));
    p.push_back(make_float3(1.000000 , 1.000000 , 1.000000 ));
    p.push_back(make_float3(1.000000 , -1.000000, 1.000000 ));
    p.push_back(make_float3(-1.000000, 1.000000 , -1.000000));
    p.push_back(make_float3(-1.000000, -1.000000, -1.000000));
    p.push_back(make_float3(-1.000000, 1.000000 , 1.000000 ));
    p.push_back(make_float3(-1.000000, -1.000000, 1.000000 ));

    n.push_back(make_float3(0.));
    n.push_back(make_float3(0.0000 , 1.0000 , 0.0000 ));
    n.push_back(make_float3(0.0000 , 0.0000 , 1.0000 ));
    n.push_back(make_float3(-1.0000, 0.0000 , 0.0000 ));
    n.push_back(make_float3(0.0000 , -1.0000, 0.0000 ));
    n.push_back(make_float3(1.0000 , 0.0000 , 0.0000 ));
    n.push_back(make_float3(0.0000 , 0.0000 , -1.0000));
    
    triangles.emplace_back(p[1], p[5], p[7], n[1], n[1], n[1]); 
    triangles.emplace_back(p[1], p[7], p[3], n[1], n[1], n[1]);
    triangles.emplace_back(p[4], p[3], p[7], n[2], n[2], n[2]); 
    triangles.emplace_back(p[4], p[7], p[8], n[2], n[2], n[2]);
    triangles.emplace_back(p[8], p[7], p[5], n[3], n[3], n[3]); 
    triangles.emplace_back(p[8], p[5], p[6], n[3], n[3], n[3]);
    triangles.emplace_back(p[6], p[2], p[4], n[4], n[4], n[4]); 
    triangles.emplace_back(p[6], p[4], p[8], n[4], n[4], n[4]);
    triangles.emplace_back(p[2], p[1], p[3], n[5], n[5], n[5]); 
    triangles.emplace_back(p[2], p[3], p[4], n[5], n[5], n[5]);
    triangles.emplace_back(p[6], p[5], p[1], n[6], n[6], n[6]); 
    triangles.emplace_back(p[6], p[1], p[2], n[6], n[6], n[6]);
}

void create_rectangle_triangles(const PropertyList& list, std::vector<Triangle>& triangles)
{
    float3 p0 = make_float3(-1, -1, 0);
    float3 p1 = make_float3( 1, -1, 0);
    float3 p2 = make_float3( 1,  1, 0);
    float3 p3 = make_float3(-1,  1, 0);
    triangles.emplace_back(p0, p1, p2);
    triangles.emplace_back(p2, p3, p0); 
}

void create_sphere_triangles(const PropertyList& list, std::vector<Triangle>& triangles)
{
    std::vector<float3> p;
    std::vector<float3> n;    

    float radius = list.getFloat("radius", 1);

    int subdiv = list.getInteger("subdiv", 10);
    int nLatitude = subdiv;
    int nLongitude = 2 * subdiv;
    float latitudeDeltaAngle = PI / nLatitude;
    float longitudeDeltaAngle = 2 * PI / nLongitude;

    p.push_back(make_float3(0, 0, radius));
    n.push_back(make_float3(0, 0, 1));
    for (int i = 1; i < nLatitude; i++) {
        float theta = i * latitudeDeltaAngle;
        for (int j = 0; j < nLongitude; j++) {
            float phi = j * longitudeDeltaAngle;
            float x = sin(theta) * cos(phi);
            float y = sin(theta) * sin(phi);
            float z = cos(theta);
            p.push_back(make_float3(x * radius, y * radius, z * radius));
            n.push_back(make_float3(x, y, z));
        }
    }
    p.push_back(make_float3(0, 0, -radius));
    n.push_back(make_float3(0, 0, -1));

    for (int i = 0; i < nLongitude; i++) {
        int a = i + 1, b = a + 1;
        if (i == nLongitude - 1) b = 1;
        triangles.push_back(Triangle(p[a], p[b], p[0], n[a], n[b], n[0]));
    }

    
    for (int i = 2; i < nLatitude; i++) {
        for (int j = 0; j < nLongitude; j++) {
            int a = (i - 1) * nLongitude + j + 1, b = a + 1, c = a - nLongitude, d = c + 1;
            if (j == nLongitude - 1) b = (i - 1) * nLongitude + 1, d = b - nLongitude;
            triangles.push_back(Triangle(p[a], p[b], p[c], n[a], n[b], n[c]));
            triangles.push_back(Triangle(p[c], p[b], p[d], n[c], n[b], n[d]));
        }
    }
    

    int bottomIdx = nLongitude * (nLatitude - 1) + 1;
    for (int i = 0; i < nLongitude; i++) {
        int a = (nLatitude - 2) * nLongitude + i + 1, b = a + 1;
        if (i == nLongitude - 1) b = (nLatitude - 2) * nLongitude + 1;
        triangles.push_back(Triangle(p[a], p[bottomIdx], p[b], n[a], n[bottomIdx], n[b]));
    }
}

ParseRecord::ParseRecord(
    const std::string filename,
    Scene* scene
    ) 
    : m_filename(filename), m_scene(scene)
{
    m_triangles = &m_scene->m_cpu_triangles;
    m_camera = &m_scene->m_camera;

    std::filesystem::path p(filename);
    m_path = p.parent_path();

    m_current_material.setZero();
    m_current_light = make_float3(0.f);

    m_tags["hide"] = EHide;
    m_tags["mode"] = EMode;
    m_tags["scene"] = EScene;

    m_tags["integrator"] = EIntegrator;
    m_tags["sensor"] = ECamera;
    m_tags["camera"] = ECamera;
    m_tags["sampler"] = ESampler;
    m_tags["film"] = EFilm;
    m_tags["rfilter"] = ERFilter;
    m_tags["emitter"] = ELight;
    m_tags["shape"] = EShape;
    m_tags["bsdf"] = EBSDF;
    m_tags["medium"] = EMedium;
    m_tags["volume"] = EVolume;
    m_tags["ref"] = ERef;

    m_tags["bool"] = EBoolean;
    m_tags["boolean"] = EBoolean;
    m_tags["integer"] = EInteger;
    m_tags["float"] = EFloat;
    m_tags["string"] = EString;
    m_tags["vector"] = EVector;
    m_tags["spectrum"] = EColor;
    m_tags["rgb"] = EColor;
    m_tags["color"] = EColor;
    m_tags["transform"] = ETransform;
    m_tags["lookat"] = ELookAt;
    m_tags["lookAt"] = ELookAt;
    m_tags["translate"] = ETranslate;
    m_tags["scale"] = EScale;
    m_tags["rotate"] = ERotate;
    m_tags["matrix"] = EMatrix;
}

void ParseRecord::createCamera(const std::string& type, const PropertyList& list)
{
    //m_camera->
}

void ParseRecord::createMaterial(const std::string& type, const PropertyList& list)
{
    m_current_material.setZero();
    if (type == "diffuse")
    {
        m_current_material.m_diffuse = list.getColor("reflectance", make_float3(0.5f));       
        m_current_material.m_type = MaterialType::MATERIAL_DIFFUSE;
    }
    else if (type == "mirror")
    {
        m_current_material.m_specular = list.getColor("reflectance", make_float3(1.f));
        m_current_material.m_ior = 100.f;
        m_current_material.m_type = MaterialType::MATERIAL_MIRROR;
    }
    else if (type == "glass")
    {
        m_current_material.m_specular = list.getColor("reflectance", make_float3(1.f));
        m_current_material.m_ior = 1.5f;
        m_current_material.m_type = MaterialType::MATERIAL_GLASS;
    }
    else if (type == "dielectric")
    {
        float int_ior = list.getFloat("intIOR", 1.5f);
        float ext_ior = list.getFloat("extIOR", 1.0f);
        float ior = int_ior / ext_ior;
        std::cout << ior << std::endl;
        m_current_material.m_specular = make_float3(1.f);
        m_current_material.m_ior = ior;
        m_current_material.m_type = MaterialType::MATERIAL_GLASS;
    }
    else if (type == "roughconductor")
    {
        m_current_material.m_diffuse = make_float3(1.0f);
        m_current_material.m_type = MaterialType::MATERIAL_DIFFUSE;
    }
    else {
        std::cout << type << std::endl;
        m_current_material.m_diffuse = make_float3(1.0f);
    }
}

void ParseRecord::createNamedMaterial(const std::string& id, const std::string& type, const PropertyList& list)
{
    createMaterial(type, list);
    m_named_material[id] = m_current_material;
}

void ParseRecord::createLight(const std::string& type, const PropertyList& list)
{
    assert(type == "area" || type == "envmap");
    if (type == "area")
    {
        m_current_light = list.getColor("radiance", make_float3(1.f));
    }
    else
    {
        std::string filename = list.getString("filename");
        filename = (m_path / filename).string();
        Transform o2w = list.getTransform("toWorld", Transform());
        m_scene->m_environment_light.setup(filename, o2w);
    }
}

void ParseRecord::createShape(const std::string& type, const PropertyList& list)
{
    assert(type == "obj" || type == "sphere" || type == "rectangle" || type == "cube");
    m_current_material.m_emission = m_current_light;

    std::vector<Triangle> triangles;
    if (type == "obj")
    {
        std::string filename = list.getString("filename");
        filename = (m_path / filename).string();
        load_obj_file(filename, triangles);
    }
    else if (type == "sphere")
    {
        create_sphere_triangles(list, triangles);
    }
    else if (type == "rectangle")
    {
        create_rectangle_triangles(list, triangles);
    }
    else if (type == "cube")
    {
        create_cube_triangles(list, triangles);
    }

    Transform transform = list.getTransform("toWorld", Transform());

    for (auto& triangle : triangles)
    {
        triangle.m_p0 = transform.transformPoint(triangle.m_p0);
        triangle.m_p1 = transform.transformPoint(triangle.m_p1);
        triangle.m_p2 = transform.transformPoint(triangle.m_p2);

        if (triangle.m_has_n)
        {
            triangle.m_n0 = transform.transformNormal(triangle.m_n0);
            triangle.m_n1 = transform.transformNormal(triangle.m_n1);
            triangle.m_n2 = transform.transformNormal(triangle.m_n2);
        }

        triangle.m_material = m_current_material;
        m_triangles->push_back(triangle);
    }
    
    m_current_material.setZero();
    m_current_light = make_float3(0.f);
}

void ParseRecord::createReference(const std::string& name, const std::string& id)
{
    m_current_material = m_named_material[id];
}

bool HasAttribute(const pugi::xml_node& node, const std::string& name) {
    for (auto& attribute : node.attributes()) {
        if (strcmp(attribute.name(), name.c_str()) == 0) {
            return true;
        }
    }
    return false;
}

std::string GetValue(const pugi::xml_node& node, const std::string& name, const std::string& defaultValue) {
    for (auto& attribute : node.attributes()) {
        if (strcmp(attribute.name(), name.c_str()) == 0) {
            return attribute.value();
        }
    }
    return defaultValue;
}

/**
 * \brief Helper function: map a position offset in bytes to a more readable line/column value
 * \param pos
 * \param record
 * \return
 */
std::string GetOffset(ptrdiff_t pos, ParseRecord& record) {
    const std::string& m_filename = record.m_filename;
    std::fstream is(m_filename);
    char buffer[1024];
    int line = 0, linestart = 0, offset = 0;
    while (is.good()) {
        is.read(buffer, sizeof(buffer));
        for (int i = 0; i < is.gcount(); ++i) {
            if (buffer[i] == '\n') {
                if (offset + i >= pos)
                    return tfm::format("line %i, col %i", line + 1, pos - linestart);
                ++line;
                linestart = offset + i;
            }
        }
        offset += (int)is.gcount();
    }
    return "byte offset " + std::to_string(pos);
}

void HandleTag(
    pugi::xml_node& node,
    PropertyList& myList,
    PropertyList& parentList,
    ParseRecord& record)
{
    auto& m_tags = record.m_tags;
    auto& m_transform = record.m_transform;

    // If isObject, add the value in children to myList, then create and add the object to fatherList
    // Otherwise, add the value to fatherList

    std::string nodeName = node.name();
    const ETag tag = m_tags[nodeName];
    const bool isObject = tag < EBoolean && tag >= EIntegrator;

    // Get name, type and id
    const std::string name = GetValue(node, "name", "");
    const std::string type = GetValue(node, "type", "");
    const std::string id = GetValue(node, "id", "");

    if (tag == EMode) {
        //RainbowRenderMode(type);
    }
    else if (isObject) {
        switch (tag) {
        case EIntegrator:
            //RainbowIntegrator(type, myList);
            break;
        case ECamera:
            record.createCamera(type, myList);
            break;
        case ESampler:
            //RainbowSampler(type, myList);
            break;
        case EFilm:
            //RainbowFilm(type, myList);
            break;
        case EShape:
            record.createShape(type, myList);
            break;
        case ELight:
            record.createLight(type, myList);
            break;
        case ERFilter:
            //RainbowFilter(type, myList);
            break;
        case EBSDF:
            if (id == "") {
                record.createMaterial(type, myList);
            }
            else {
                record.createNamedMaterial(id, type, myList);
            }
            break;
        case EMedium: {
            //if (id == "") {
            //    RainbowMedium(type, name, myList);
            //}
            //else {
            //    RainbowNamedMedium(id, type, name, myList);
            //}
            break;
        }
        case EVolume:
            //RainbowVolume(type, name, myList);
            break;
        case ERef:
            record.createReference(name, id);
            break;
        }
    }
    else {
        std::string value = GetValue(node, "value", "");
        switch (tag) {
        case EBoolean: {
            parentList.setBoolean(name, toBoolean(value));
            break;
        }
        case EInteger: {
            parentList.setInteger(name, toInteger(value));
            break;
        }
        case EFloat: {
            parentList.setFloat(name, toFloat(value));
            break;
        }
        case EString: {
            parentList.setString(name, value);
            break;
        }
        case EVector: {
            parentList.setVector(name, toVector(value));
            break;
        }
        case EColor: {
            if (strcmp(node.name(), "spectrum") == 0) {
                // TODO: Fix Spectrum declared with wavelength
                assert(false, "No implemented!");
            }
            else if (strcmp(node.name(), "rgb") == 0 || strcmp(node.name(), "color") == 0) {
                parentList.setColor(name, toColor(value));
            }
            break;
        }
        case ETransform: {
            parentList.setTransform(name, m_transform);
            break;
        }
        case ELookAt: {
            const float3 target = toVector(node.attribute("target").value());
            const float3 origin = toVector(node.attribute("origin").value());
            const float3 up = toVector(node.attribute("up").value());
            m_transform *= LookAt(target, origin, up);
            break;
        }
        case ETranslate: {
            if (value == "") {
                const float x = toFloat(GetValue(node, "x", "0"));
                const float y = toFloat(GetValue(node, "y", "0"));
                const float z = toFloat(GetValue(node, "z", "0"));
                m_transform *= Translate(make_float3(x, y, z));
            }
            else {
                const float3 delta = toVector(value);
                m_transform *= Translate(delta);
            }
            break;
        }
        case EScale: {
            if (value == "") {
                const float x = toFloat(GetValue(node, "x", "1"));
                const float y = toFloat(GetValue(node, "y", "1"));
                const float z = toFloat(GetValue(node, "z", "1"));
                m_transform *= Scale(make_float3(x, y, z));
            }
            else {
                const float3 scale = toVector(value);
                m_transform *= Scale(scale);
            }
            break;
        }
        case ERotate: {
            const float x = toFloat(GetValue(node, "x", "0"));
            const float y = toFloat(GetValue(node, "y", "0"));
            const float z = toFloat(GetValue(node, "z", "0"));
            const float angle = toFloat(GetValue(node, "angle", "0"));
            m_transform *= Rotate(angle, make_float3(x, y, z));
            break;
        }
        case EMatrix: {
            const Matrix4x4 mat = toMatrix(value);
            m_transform *= Transform(mat);
            break;
        }
        }
    }
}

void ParseTag(pugi::xml_node& node, PropertyList& parentList, ParseRecord& record) {
    auto& m_tags = record.m_tags;
    auto& m_transform = record.m_transform;

    // Skip the comment and other information like version and encoding.
    if (node.type() == pugi::node_comment ||
        node.type() == pugi::node_declaration) return;
    // Check the name of tag
    assert(m_tags.count(node.name()) == 1, std::string("Unexpected type ") + node.name());
    const ETag tag = m_tags[node.name()];

    // If isObject, add the value in children to myList, then create and add the object to fatherList
    // But EShape isn't added into fatherList, it is added into RenderOptions
    // Otherwise, add the value to fatherList

    // Initialize transform
    if (tag == ETransform) {
        m_transform.Identify();
    }
    else if (tag == EHide) {
        return;
    }

    // Add children nodes' value/object to myList
    PropertyList myList;
    for (pugi::xml_node& child : node.children()) {
        ParseTag(child, myList, record);
    }

    // Handle tags
    HandleTag(node, myList, parentList, record);
}

void Parse(ParseRecord& record) {
    const auto& m_filename = record.m_filename;
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(m_filename.c_str());

    assert(result, "Error while parsing \"" + m_filename + "\": " +
        result.description() + " (at " + GetOffset(result.offset, record) + ")");

    //RainbowSceneBegin();
    PropertyList list;
    ParseTag(*doc.begin(), list, record);
    //RainbowSceneEnd();
}