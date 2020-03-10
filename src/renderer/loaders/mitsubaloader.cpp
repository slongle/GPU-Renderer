#include "mitsubaloader.h"
#include "renderer/scene.h"
#include "renderer/loaders/objloader.h"

void load_mitsuba_file(
    const std::string& filename, 
    Scene* scene)
{
    ParseRecord record(
        filename, 
        scene);
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
    ret.r = strtof(str.c_str(), &endptr); endptr++;
    ret.g = strtof(endptr, &endptr); endptr++;
    ret.b = strtof(endptr, &endptr);
    return ret;
}

void create_cube_triangles(
    const PropertyList& list, 
    TriangleMesh* mesh)
{
    std::vector<float3> p;
    std::vector<float3> n;
    std::vector<float2> uv;
    std::vector<int3> index;

    p.push_back(make_float3(1.000000 , 1.000000 , -1.000000));        
    p.push_back(make_float3(1.000000 , -1.000000, -1.000000));
    p.push_back(make_float3(1.000000 , 1.000000 , 1.000000 ));
    p.push_back(make_float3(1.000000 , -1.000000, 1.000000 ));
    p.push_back(make_float3(-1.000000, 1.000000 , -1.000000));
    p.push_back(make_float3(-1.000000, -1.000000, -1.000000));
    p.push_back(make_float3(-1.000000, 1.000000 , 1.000000 ));
    p.push_back(make_float3(-1.000000, -1.000000, 1.000000 ));
    
    n.push_back(make_float3(0.0000 , 1.0000 , 0.0000 ));
    n.push_back(make_float3(0.0000 , 0.0000 , 1.0000 ));
    n.push_back(make_float3(-1.0000, 0.0000 , 0.0000 ));
    n.push_back(make_float3(0.0000 , -1.0000, 0.0000 ));
    n.push_back(make_float3(1.0000 , 0.0000 , 0.0000 ));
    n.push_back(make_float3(0.0000 , 0.0000 , -1.0000));
    
    uv.push_back(make_float2(0.625000, 0.500000));
    uv.push_back(make_float2(0.875000, 0.500000));
    uv.push_back(make_float2(0.875000, 0.750000));
    uv.push_back(make_float2(0.625000, 0.750000));
    uv.push_back(make_float2(0.375000, 0.750000));
    uv.push_back(make_float2(0.625000, 1.000000));
    uv.push_back(make_float2(0.375000, 1.000000));
    uv.push_back(make_float2(0.375000, 0.000000));
    uv.push_back(make_float2(0.625000, 0.000000));
    uv.push_back(make_float2(0.625000, 0.250000));
    uv.push_back(make_float2(0.375000, 0.250000));
    uv.push_back(make_float2(0.125000, 0.500000));
    uv.push_back(make_float2(0.375000, 0.500000));
    uv.push_back(make_float2(0.125000, 0.750000));

    int x[] = { 0, 0, 0, 4, 1, 0, 6, 2, 0, 2, 3, 0, 3, 4, 1, 2, 3, 1, 6, 5, 1, 7, 6, 1, 7, 7, 2, 6, 8, 2, 4, 9, 2, 5, 10, 2, 5, 11, 3, 1, 12, 3, 3, 4, 3, 7, 13, 3, 1, 12, 4, 0, 0, 4, 2, 3, 4, 3, 4, 4, 5, 10, 5, 4, 9, 5, 0, 0, 5, 1, 12, 5 };
    for (uint32 i = 0; i < 6; i++)
    {
        int3 quad[4];
        for (uint32 j = 0; j < 4; j++) 
        {
            uint32 idx = i * 12 + j * 3;
            quad[j] = make_int3(x[idx + 0], x[idx + 1], x[idx + 2]);
        }
        index.push_back(quad[0]);
        index.push_back(quad[1]);
        index.push_back(quad[2]);
        index.push_back(quad[0]);
        index.push_back(quad[2]);
        index.push_back(quad[3]);
    }

    mesh->m_triangle_num = 12;
    mesh->m_cpu_p = p;
    mesh->m_cpu_n = n;
    mesh->m_cpu_uv = uv;
    mesh->m_cpu_index = index;
}

void create_rectangle_triangles(
    const PropertyList& list, 
    TriangleMesh* mesh)
{
    std::vector<float3> p;
    std::vector<float3> n;
    std::vector<float2> uv;
    std::vector<int3> index;

    p.push_back(make_float3(-1, -1, 0));
    p.push_back(make_float3(1, -1, 0));
    p.push_back(make_float3(1, 1, 0));
    p.push_back(make_float3(-1, 1, 0));

    uv.push_back(make_float2(0, 0));
    uv.push_back(make_float2(1, 0));
    uv.push_back(make_float2(1, 1));
    uv.push_back(make_float2(0, 1));

    n.push_back(make_float3(0, 0, 1));

    index.push_back(make_int3(0, 0, 0));
    index.push_back(make_int3(1, 1, 0));
    index.push_back(make_int3(2, 2, 0));
    index.push_back(make_int3(2, 2, 0));
    index.push_back(make_int3(3, 3, 0));
    index.push_back(make_int3(0, 0, 0));
    

    mesh->m_triangle_num = 2;
    mesh->m_cpu_p = p;
    mesh->m_cpu_n = n;
    mesh->m_cpu_uv = uv;
    mesh->m_cpu_index = index;
}

void create_sphere_triangles(
    const PropertyList& list, 
    TriangleMesh* mesh)
{
    std::vector<float3> p;
    std::vector<float3> n;   
    std::vector<float2> uv;
    std::vector<int3> index;

    float radius = list.getFloat("radius", 1);

    int subdiv = list.getInteger("subdiv", 30);
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
        index.push_back(make_int3(a, -1, a));
        index.push_back(make_int3(b, -1, b));
        index.push_back(make_int3(0, -1, 0));
    }
    
    for (int i = 2; i < nLatitude; i++) {
        for (int j = 0; j < nLongitude; j++) {
            int a = (i - 1) * nLongitude + j + 1, b = a + 1, c = a - nLongitude, d = c + 1;
            if (j == nLongitude - 1) b = (i - 1) * nLongitude + 1, d = b - nLongitude;
            index.push_back(make_int3(a, -1, a));
            index.push_back(make_int3(b, -1, b));
            index.push_back(make_int3(c, -1, c));
            index.push_back(make_int3(c, -1, c));
            index.push_back(make_int3(b, -1, b));
            index.push_back(make_int3(d, -1, d));
        }
    }
    
    int bottomIdx = nLongitude * (nLatitude - 1) + 1;
    for (int i = 0; i < nLongitude; i++) {
        int a = (nLatitude - 2) * nLongitude + i + 1, b = a + 1;
        if (i == nLongitude - 1) b = (nLatitude - 2) * nLongitude + 1;
        index.push_back(make_int3(a, -1, a));
        index.push_back(make_int3(bottomIdx, -1, bottomIdx));
        index.push_back(make_int3(b, -1, b));
    }

    mesh->m_triangle_num = (nLatitude - 1) * nLongitude * 2; 
    mesh->m_cpu_p = p;
    mesh->m_cpu_n = n;
    mesh->m_cpu_uv = uv;
    mesh->m_cpu_index = index;
}

ParseRecord::ParseRecord(
    const std::string filename,
    Scene* scene
    ) 
    : m_filename(filename), m_scene(scene)
{
    m_triangles = &m_scene->m_cpu_triangles;
    m_camera = &m_scene->m_camera;
    m_frame_buffer = &m_scene->m_frame_buffer;

    std::filesystem::path p(filename);
    m_path = p.parent_path();

    m_current_light = Spectrum(0.f);

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
    m_tags["texture"] = ETexture;
}

void ParseRecord::createCamera(const std::string& type, const PropertyList& list)
{
    Transform c2w = list.getTransform("toWorld", Transform());
    float fov = list.getFloat("fov", 30.f);
    bool free = list.getBoolean("free", false);
    m_camera->setup(c2w, fov, m_frame_buffer->m_resolution_x, m_frame_buffer->m_resolution_y, free);            
}

void ParseRecord::createFilm(const std::string& type, const PropertyList& list)
{
    uint32 width = list.getInteger("width", 256);
    uint32 height = list.getInteger("height", 256);
    m_frame_buffer->resize(width, height);
}

void ParseRecord::createMaterial(const std::string& type, const PropertyList& list)
{
    if (type == "diffuse")
    {
        std::shared_ptr<Texture> color = list.getTexture("reflectance", Spectrum(0.3f));
        m_scene->m_textures.push_back(color);

        m_current_material.m_color = color->view();
        m_current_material.m_type = MaterialType::MATERIAL_DIFFUSE;
    }
    else if (type == "mirror")
    {
        std::shared_ptr<Texture> color = list.getTexture("reflectance", Spectrum(1.0f));
        std::shared_ptr<Texture> ior = list.getTexture("ior", 100.f);
        m_scene->m_textures.push_back(color);
        m_scene->m_textures.push_back(ior);

        m_current_material.m_color = color->view();
        m_current_material.m_ior = ior->view();
        m_current_material.m_type = MaterialType::MATERIAL_SPECULAR;
    }
    else if (type == "glass")
    {
        std::shared_ptr<Texture> color = list.getTexture("reflectance", Spectrum(1.0f));
        std::shared_ptr<Texture> ior = list.getTexture("ior", 1.5f);
        m_scene->m_textures.push_back(color);
        m_scene->m_textures.push_back(ior);

        m_current_material.m_color = color->view();
        m_current_material.m_ior = ior->view();
        m_current_material.m_type = MaterialType::MATERIAL_SPECULAR;
    }
    else if (type == "roughdielectric" || type == "dielectric")
    {
        std::shared_ptr<Texture> color = list.getTexture("reflectance", Spectrum(1.0f));
        float ior;
        if (list.findFloat("intIOR"))
        {
            float int_ior = list.getFloat("intIOR", 1.5f);
            float ext_ior = list.getFloat("extIOR", 1.0f);
            ior = int_ior / ext_ior;            
        }
        else
        {
            ior = list.getFloat("ior", 1.5f);
        }
        std::shared_ptr<Texture> ior_texture = list.getTexture("ior", 1.5f);
        m_scene->m_textures.push_back(color);
        m_scene->m_textures.push_back(ior_texture);

        m_current_material.m_color = color->view();
        m_current_material.m_ior = ior_texture->view();
        m_current_material.m_type = MaterialType::MATERIAL_SPECULAR;
    }
    else if (type == "roughconductor" || type == "conductor")
    {
        std::shared_ptr<Texture> color = list.getTexture("specularReflectance", Spectrum(0.3f));
        std::shared_ptr<Texture> etaI, etaT;
        if (list.findColor("etaI"))
        {
            etaI = list.getTexture("etaI", Spectrum(1.f)); 
            etaT = list.getTexture("etaT", Spectrum(0.200438, 0.924033, 1.10221));
        }
        else
        {
            etaI = list.getTexture("etaI", Spectrum(1.f));
            etaT = list.getTexture("eta", Spectrum(0.200438, 0.924033, 1.10221));            
        }
        std::shared_ptr<Texture> k = list.getTexture("k", Spectrum(3.91295, 2.45285, 2.14219));
        float alphaU, alphaV;
        if (list.findFloat("alphaU"))
        {
            alphaU = list.getFloat("alphaU", 0.1);
            alphaV = list.getFloat("alphaV", 0.1);
        }
        else 
        {
            alphaU = alphaV = list.getFloat("alpha", 0.1);
        }
        std::shared_ptr<Texture> alphaU_texture = list.getTexture("alphaU", alphaU);
        std::shared_ptr<Texture> alphaV_texture = list.getTexture("alphaU", alphaV);

        m_scene->m_textures.push_back(color);
        m_scene->m_textures.push_back(etaI);
        m_scene->m_textures.push_back(etaT);
        m_scene->m_textures.push_back(k);
        m_scene->m_textures.push_back(alphaU_texture);
        m_scene->m_textures.push_back(alphaV_texture);

        m_current_material.m_color = color->view();
        m_current_material.m_etaI = etaI->view();
        m_current_material.m_etaT = etaT->view();
        m_current_material.m_k = k->view();
        m_current_material.m_alpha_x = alphaU_texture->view();
        m_current_material.m_alpha_y = alphaV_texture->view();
        m_current_material.m_type = MaterialType::MATERIAL_ROUGH_CONDUCTOR;
    }
    else if (type == "roughplastic")
    {
        std::shared_ptr<Texture> etaI, etaT;
        if (list.findColor("etaI"))
        {
            etaI = list.getTexture("etaI", Spectrum(1.5f));
            etaT = list.getTexture("etaT", Spectrum(1.0f));
        }
        else
        {
            etaI = list.getTexture("eta", Spectrum(1.5f));
            etaT = list.getTexture("etaI", Spectrum(1.0f));
        }
        float alphaU, alphaV;
        if (list.findFloat("alphaU"))
        {
            alphaU = list.getFloat("alphaU", 0.1);
            alphaV = list.getFloat("alphaV", 0.1);
        }
        else
        {
            alphaU = alphaV = list.getFloat("alpha", 0.1);
        }
        std::shared_ptr<Texture> alphaU_texture = list.getTexture("alphaU", alphaU);
        std::shared_ptr<Texture> alphaV_texture = list.getTexture("alphaU", alphaV);
        std::shared_ptr<Texture> color0 = list.getTexture("diffuseReflectance", Spectrum(0.5f));
        std::shared_ptr<Texture> color1 = list.getTexture("specularReflectance", Spectrum(1.0f));

        m_scene->m_textures.push_back(color0);
        m_scene->m_textures.push_back(color1);
        m_scene->m_textures.push_back(etaI);
        m_scene->m_textures.push_back(etaT);
        m_scene->m_textures.push_back(alphaU_texture);
        m_scene->m_textures.push_back(alphaV_texture);

        m_current_material.m_color = color0->view(); // Diffuse
        m_current_material.m_color1 = color1->view(); // Specular
        m_current_material.m_etaI = etaI->view();
        m_current_material.m_etaT = etaT->view();
        m_current_material.m_alpha_x = alphaU_texture->view();
        m_current_material.m_alpha_y = alphaV_texture->view();
        m_current_material.m_type = MaterialType::MATERIAL_ROUGH_PLASTIC;
    }
    else if (type == "disney")
    {
        Spectrum color = list.getColor("color", Spectrum(0.5f));
        float metallic = list.getFloat("metallic", 0.f);
        float eta = list.getFloat("eta", 1.5f);
        float roughness = list.getFloat("roughness", 0.5f);
    }
    else {
        std::cout << type << std::endl;
        m_current_material.m_color = list.getTexture("Unknown", Spectrum(0.2f))->view();
        m_current_material.m_type = MaterialType::MATERIAL_DIFFUSE;
    }
}

void ParseRecord::createNamedMaterial(
    const std::string& id, 
    const std::string& type, 
    const PropertyList& list)
{
    createMaterial(type, list);
    m_named_material[id] = m_current_material;
}

void ParseRecord::createNamedMaterial(
    const std::string& id)
{
    m_named_material[id] = m_current_material;
}

void ParseRecord::createLight(const std::string& type, const PropertyList& list)
{
    assert(type == "area" || type == "envmap" || type == "constant");
    if (type == "area")
    {
        float scale = list.getFloat("scale", 1.f);
        Spectrum radiance = list.getColor("radiance", Spectrum(1.f));
        m_current_light = radiance * scale;
    }
    else if (type == "constant")
    {
        Spectrum r = list.getColor("radiance", Spectrum(1.f));
        m_scene->m_environment_light.setup(r);
    }
    else if (type == "envmap")
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
    
    TriangleMesh mesh;

    if (type == "obj")
    {
        std::string filename = list.getString("filename");
        //std::cout << filename << std::endl;
        filename = (m_path / filename).string();
        load_obj_file(filename, &mesh);        
    }
    else if (type == "sphere")
    {
        create_sphere_triangles(list, &mesh);
    }
    else if (type == "rectangle")
    {
        create_rectangle_triangles(list, &mesh);
    }
    else if (type == "cube")
    {
        create_cube_triangles(list, &mesh);
    }

    Transform o2w = list.getTransform("toWorld", Transform());
    for (uint32 i = 0; i < mesh.m_cpu_p.size(); i++)
    {
        mesh.m_cpu_p[i] = o2w.transformPoint(mesh.m_cpu_p[i]);
    }
    for (uint32 i = 0; i < mesh.m_cpu_n.size(); i++)
    {
        mesh.m_cpu_n[i] = normalize(o2w.transformNormal(mesh.m_cpu_n[i]));
    }
    mesh.m_material = m_current_material;    

    m_scene->m_meshes.push_back(mesh);

    m_current_light = Spectrum(0.f);
}

void ParseRecord::createReference(const std::string& name, const std::string& id)
{
    m_current_material = m_named_material[id];
}

std::shared_ptr<Texture>
ParseRecord::createTexture(const std::string& type, const PropertyList& list)
{    
    assert(type == "bitmap" || type == "checkerboard");
    float uoffset = list.getFloat("uoffset", 0.f);
    float voffset = list.getFloat("voffset", 0.f);
    float uscale = list.getFloat("uscale", 1.f);
    float vscale = list.getFloat("vscale", 1.f);
    float2 uvoffset = make_float2(uoffset, voffset);
    float2 uvscale = make_float2(uscale, vscale);

    Texture* ptr = nullptr;
    if (type == "bitmap")
    {        
        std::string filename = list.getString("filename");
        filename = (m_path / filename).string();
        ptr = new Texture(filename, uvoffset, uvscale);
    }
    else
    {
        Spectrum color0 = list.getColor("color0", Spectrum(0.4f));
        Spectrum color1 = list.getColor("color1", Spectrum(0.2f));
        ptr = new Texture(color0, color1, uvoffset, uvscale);
    }
    return std::shared_ptr<Texture>(ptr);
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
            record.createFilm(type, myList);
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
            if (type == "twosided")
            {
                if (id == "") {
                }
                else {
                    record.createNamedMaterial(id);
                }
            }
            else
            {
                if (id == "") {
                    record.createMaterial(type, myList);
                }
                else {
                    record.createNamedMaterial(id, type, myList);
                }
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
        case ETexture: {
            parentList.setTexture(name, record.createTexture(type, myList));
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