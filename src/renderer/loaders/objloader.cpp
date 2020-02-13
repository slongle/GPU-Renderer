#include "objloader.h"
#include "renderer/triangle.h"
#include "renderer/scene.h"

#include <iostream>
#include <filesystem>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"


bool load_obj_file(
    const std::string& filename, 
    TriangleMesh* mesh)
{
    std::filesystem::path filePath(filename);
    std::string basePath(filePath.parent_path().string());

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str(), basePath.c_str());

    if (!warn.empty()) {
        //std::cout << "WARN: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << "ERR: " << err << std::endl;
    }

    if (!ret) {
        std::cout << "Failed to load/parse .obj.\n";
        return false;
    }

    /*
    std::cout << "# of vertices  : " << (attrib.vertices.size() / 3) << std::endl;
    std::cout << "# of normals   : " << (attrib.normals.size() / 3) << std::endl;
    std::cout << "# of texcoords : " << (attrib.texcoords.size() / 2) << std::endl;

    std::cout << "# of shapes    : " << shapes.size() << std::endl;
    std::cout << "# of materials : " << materials.size() << std::endl;
    */     

    std::vector<float3> vertices;
    for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
        vertices.push_back(make_float3(
            static_cast<const float>(attrib.vertices[3 * v + 0]),
            static_cast<const float>(attrib.vertices[3 * v + 1]),
            static_cast<const float>(attrib.vertices[3 * v + 2])));
    }

    std::vector<float3> normals;
    for (size_t v = 0; v < attrib.normals.size() / 3; v++) {
        normals.push_back(make_float3(
            static_cast<const double>(attrib.normals[3 * v + 0]),
            static_cast<const double>(attrib.normals[3 * v + 1]),
            static_cast<const double>(attrib.normals[3 * v + 2])));
    }

    std::vector<float2> uvs;
    for (size_t v = 0; v < attrib.texcoords.size() / 2; v++) {
            uvs.push_back(make_float2(
                static_cast<const double>(attrib.texcoords[2 * v + 0]),
                static_cast<const double>(attrib.texcoords[2 * v + 1])));
    }
    
    std::vector<int3> indices;
    uint32 nTriangles = 0;
    // For each shape
    for (size_t i = 0; i < shapes.size(); i++) {

        size_t index_offset = 0;

        assert(shapes[i].mesh.num_face_vertices.size() ==
            shapes[i].mesh.material_ids.size());

        assert(shapes[i].mesh.num_face_vertices.size() ==
            shapes[i].mesh.smoothing_group_ids.size());

        // For each face
        for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
            size_t fnum = shapes[i].mesh.num_face_vertices[f];
            assert(fnum == 3);

            indices.push_back(make_int3(
                shapes[i].mesh.indices[index_offset + 0].vertex_index,
                shapes[i].mesh.indices[index_offset + 0].normal_index,
                shapes[i].mesh.indices[index_offset + 0].texcoord_index));

            indices.push_back(make_int3(
                shapes[i].mesh.indices[index_offset + 1].vertex_index,
                shapes[i].mesh.indices[index_offset + 1].normal_index,
                shapes[i].mesh.indices[index_offset + 1].texcoord_index));

            indices.push_back(make_int3(
                shapes[i].mesh.indices[index_offset + 2].vertex_index,
                shapes[i].mesh.indices[index_offset + 2].normal_index,
                shapes[i].mesh.indices[index_offset + 2].texcoord_index));

            nTriangles++;
            index_offset += fnum;
        }
    }

    mesh->m_triangle_num = nTriangles;
    mesh->m_cpu_p = vertices;
    mesh->m_cpu_n = normals;
    mesh->m_cpu_uv = uvs;
    mesh->m_cpu_index = indices;
    //mesh->m_cpu_p.copyFrom(vertices.size(), HOST_BUFFER, vertices.data());
    //mesh->m_cpu_n.copyFrom(normals.size(), HOST_BUFFER, normals.data());
    //mesh->m_cpu_uv.copyFrom(uvs.size(), HOST_BUFFER, uvs.data());
    //mesh->m_cpu_index.copyFrom(indices.size(), HOST_BUFFER, indices.data());

    return true;
}


bool load_obj_mtl_file(
    const std::string& filename, 
    Scene* scene)
{
    std::filesystem::path filePath(filename);
    std::string basePath(filePath.parent_path().string());

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str(),basePath.c_str());

    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << "ERR: " << err << std::endl;
    }

    if (!ret) {
        std::cout << "Failed to load/parse .obj.\n";
        return false;
    }

    /*
    std::cout << "# of vertices  : " << (attrib.vertices.size() / 3) << std::endl;
    std::cout << "# of normals   : " << (attrib.normals.size() / 3) << std::endl;
    std::cout << "# of texcoords : " << (attrib.texcoords.size() / 2) << std::endl;

    std::cout << "# of shapes    : " << shapes.size() << std::endl;
    std::cout << "# of materials : " << materials.size() << std::endl;
    */

    /*
    for (size_t i = 0; i < materials.size(); i++) {
        printf("material[%ld].name = %s\n", static_cast<long>(i),
            materials[i].name.c_str());
        printf("  material.Ka = (%f, %f ,%f)\n",
            static_cast<const double>(materials[i].ambient[0]),
            static_cast<const double>(materials[i].ambient[1]),
            static_cast<const double>(materials[i].ambient[2]));
        printf("  material.Kd = (%f, %f ,%f)\n",
            static_cast<const double>(materials[i].diffuse[0]),
            static_cast<const double>(materials[i].diffuse[1]),
            static_cast<const double>(materials[i].diffuse[2]));
        printf("  material.Ks = (%f, %f ,%f)\n",
            static_cast<const double>(materials[i].specular[0]),
            static_cast<const double>(materials[i].specular[1]),
            static_cast<const double>(materials[i].specular[2]));
        printf("  material.Tr = (%f, %f ,%f)\n",
            static_cast<const double>(materials[i].transmittance[0]),
            static_cast<const double>(materials[i].transmittance[1]),
            static_cast<const double>(materials[i].transmittance[2]));
        printf("  material.Ke = (%f, %f ,%f)\n",
            static_cast<const double>(materials[i].emission[0]),
            static_cast<const double>(materials[i].emission[1]),
            static_cast<const double>(materials[i].emission[2]));
        printf("  material.Ns = %f\n",
            static_cast<const double>(materials[i].shininess));
        printf("  material.Ni = %f\n", static_cast<const double>(materials[i].ior));
        printf("  material.dissolve = %f\n",
            static_cast<const double>(materials[i].dissolve));
        printf("  material.illum = %d\n", materials[i].illum);
        printf("  material.map_Ka = %s\n", materials[i].ambient_texname.c_str());
        printf("  material.map_Kd = %s\n", materials[i].diffuse_texname.c_str());
        printf("  material.map_Ks = %s\n", materials[i].specular_texname.c_str());
        printf("  material.map_Ns = %s\n",
            materials[i].specular_highlight_texname.c_str());
        printf("  material.map_bump = %s\n", materials[i].bump_texname.c_str());
        printf("    bump_multiplier = %f\n", static_cast<const double>(materials[i].bump_texopt.bump_multiplier));
        printf("  material.map_d = %s\n", materials[i].alpha_texname.c_str());
        printf("  material.disp = %s\n", materials[i].displacement_texname.c_str());
        printf("  <<PBR>>\n");
        printf("  material.Pr     = %f\n", static_cast<const double>(materials[i].roughness));
        printf("  material.Pm     = %f\n", static_cast<const double>(materials[i].metallic));
        printf("  material.Ps     = %f\n", static_cast<const double>(materials[i].sheen));
        printf("  material.Pc     = %f\n", static_cast<const double>(materials[i].clearcoat_thickness));
        printf("  material.Pcr    = %f\n", static_cast<const double>(materials[i].clearcoat_thickness));
        printf("  material.aniso  = %f\n", static_cast<const double>(materials[i].anisotropy));
        printf("  material.anisor = %f\n", static_cast<const double>(materials[i].anisotropy_rotation));
        printf("  material.map_Ke = %s\n", materials[i].emissive_texname.c_str());
        printf("  material.map_Pr = %s\n", materials[i].roughness_texname.c_str());
        printf("  material.map_Pm = %s\n", materials[i].metallic_texname.c_str());
        printf("  material.map_Ps = %s\n", materials[i].sheen_texname.c_str());
        printf("  material.norm   = %s\n", materials[i].normal_texname.c_str());
        std::map<std::string, std::string>::const_iterator it(
            materials[i].unknown_parameter.begin());
        std::map<std::string, std::string>::const_iterator itEnd(
            materials[i].unknown_parameter.end());

        for (; it != itEnd; it++) {
            printf("  material.%s = %s\n", it->first.c_str(), it->second.c_str());
        }
        printf("\n");
    }
    */

    std::vector<Material> mats;
    for (size_t i = 0; i < materials.size(); i++) {
        float ior = 0;
        if (materials[i].illum >= 5) {
            ior = materials[i].ior;
        }
        Spectrum diffuse = Spectrum(
            static_cast<const float>(materials[i].diffuse[0]),
            static_cast<const float>(materials[i].diffuse[1]),
            static_cast<const float>(materials[i].diffuse[2]));
        Spectrum specular = Spectrum(
            static_cast<const float>(materials[i].specular[0]),
            static_cast<const float>(materials[i].specular[1]),
            static_cast<const float>(materials[i].specular[2]));
        Spectrum emission = Spectrum(
            static_cast<const float>(materials[i].emission[0]),
            static_cast<const float>(materials[i].emission[1]),
            static_cast<const float>(materials[i].emission[2]));
        mats.emplace_back(diffuse, specular, emission, ior);
    }

    /*
    for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {        
        printf("  v[%ld] = (%f, %f, %f)\n", static_cast<long>(v),
            static_cast<const double>(attrib.vertices[3 * v + 0]),
            static_cast<const double>(attrib.vertices[3 * v + 1]),
            static_cast<const double>(attrib.vertices[3 * v + 2]));        
    }

    for (size_t v = 0; v < attrib.normals.size() / 3; v++) {
        printf("  n[%ld] = (%f, %f, %f)\n", static_cast<long>(v),
            static_cast<const double>(attrib.normals[3 * v + 0]),
            static_cast<const double>(attrib.normals[3 * v + 1]),
            static_cast<const double>(attrib.normals[3 * v + 2]));
    }

    for (size_t v = 0; v < attrib.texcoords.size() / 2; v++) {
        printf("  uv[%ld] = (%f, %f)\n", static_cast<long>(v),
            static_cast<const double>(attrib.texcoords[2 * v + 0]),
            static_cast<const double>(attrib.texcoords[2 * v + 1]));
    }
    */

    /*
    // For each shape
    for (size_t i = 0; i < shapes.size(); i++) {
        printf("shape[%ld].name = %s\n", static_cast<long>(i),
            shapes[i].name.c_str());
        printf("Size of shape[%ld].mesh.indices: %lu\n", static_cast<long>(i),
            static_cast<unsigned long>(shapes[i].mesh.indices.size()));

        size_t index_offset = 0;

        assert(shapes[i].mesh.num_face_vertices.size() ==
            shapes[i].mesh.material_ids.size());

        assert(shapes[i].mesh.num_face_vertices.size() ==
            shapes[i].mesh.smoothing_group_ids.size());

        printf("shape[%ld].num_faces: %lu\n", static_cast<long>(i),
            static_cast<unsigned long>(shapes[i].mesh.num_face_vertices.size()));

        // For each face
        for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
            size_t fnum = shapes[i].mesh.num_face_vertices[f];

            printf("  face[%ld].fnum = %ld\n", static_cast<long>(f),
                static_cast<unsigned long>(fnum));

            // For each vertex in the face
            for (size_t v = 0; v < fnum; v++) {
                tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + v];
                printf("    face[%ld].v[%ld].idx = %d/%d/%d\n", static_cast<long>(f),
                    static_cast<long>(v), idx.vertex_index, idx.normal_index,
                    idx.texcoord_index);
            }

            printf("  face[%ld].material_id = %d\n", static_cast<long>(f),
                shapes[i].mesh.material_ids[f]);
            printf("  face[%ld].smoothing_group_id = %d\n", static_cast<long>(f),
                shapes[i].mesh.smoothing_group_ids[f]);

            index_offset += fnum;
        }

        printf("shape[%ld].num_tags: %lu\n", static_cast<long>(i),
            static_cast<unsigned long>(shapes[i].mesh.tags.size()));
        for (size_t t = 0; t < shapes[i].mesh.tags.size(); t++) {
            printf("  tag[%ld] = %s ", static_cast<long>(t),
                shapes[i].mesh.tags[t].name.c_str());
            printf(" ints: [");
            for (size_t j = 0; j < shapes[i].mesh.tags[t].intValues.size(); ++j) {
                printf("%ld", static_cast<long>(shapes[i].mesh.tags[t].intValues[j]));
                if (j < (shapes[i].mesh.tags[t].intValues.size() - 1)) {
                    printf(", ");
                }
            }
            printf("]");

            printf(" floats: [");
            for (size_t j = 0; j < shapes[i].mesh.tags[t].floatValues.size(); ++j) {
                printf("%f", static_cast<const double>(
                    shapes[i].mesh.tags[t].floatValues[j]));
                if (j < (shapes[i].mesh.tags[t].floatValues.size() - 1)) {
                    printf(", ");
                }
            }
            printf("]");

            printf(" strings: [");
            for (size_t j = 0; j < shapes[i].mesh.tags[t].stringValues.size(); ++j) {
                printf("%s", shapes[i].mesh.tags[t].stringValues[j].c_str());
                if (j < (shapes[i].mesh.tags[t].stringValues.size() - 1)) {
                    printf(", ");
                }
            }
            printf("]");
            printf("\n");
        }
    }
    */

    std::vector<float3> vertices;
    for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
        vertices.push_back(make_float3(
            static_cast<const float>(attrib.vertices[3 * v + 0]),
            static_cast<const float>(attrib.vertices[3 * v + 1]),
            static_cast<const float>(attrib.vertices[3 * v + 2])));
    }

    std::vector<float3> normals;
    for (size_t v = 0; v < attrib.normals.size() / 3; v++) {
        normals.push_back(make_float3(
            static_cast<const double>(attrib.normals[3 * v + 0]),
            static_cast<const double>(attrib.normals[3 * v + 1]),
            static_cast<const double>(attrib.normals[3 * v + 2])));
    }

    std::vector<float2> uvs;
    for (size_t v = 0; v < attrib.texcoords.size() / 2; v++) {
            uvs.push_back(make_float2(
                static_cast<const double>(attrib.texcoords[2 * v + 0]),
                static_cast<const double>(attrib.texcoords[2 * v + 1])));
    }
    
    /*std::vector<int3> indices;
    uint32 nTriangles = 0;
    // For each shape
    for (size_t i = 0; i < shapes.size(); i++) {

        size_t index_offset = 0;

        assert(shapes[i].mesh.num_face_vertices.size() ==
            shapes[i].mesh.material_ids.size());

        assert(shapes[i].mesh.num_face_vertices.size() ==
            shapes[i].mesh.smoothing_group_ids.size());

        // For each face
        for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
            size_t fnum = shapes[i].mesh.num_face_vertices[f];
            assert(fnum == 3);

            indices.push_back(make_int3(
                shapes[i].mesh.indices[index_offset + 0].vertex_index,
                shapes[i].mesh.indices[index_offset + 0].normal_index,
                shapes[i].mesh.indices[index_offset + 0].texcoord_index));

            indices.push_back(make_int3(
                shapes[i].mesh.indices[index_offset + 1].vertex_index,
                shapes[i].mesh.indices[index_offset + 1].normal_index,
                shapes[i].mesh.indices[index_offset + 1].texcoord_index));

            indices.push_back(make_int3(
                shapes[i].mesh.indices[index_offset + 2].vertex_index,
                shapes[i].mesh.indices[index_offset + 2].normal_index,
                shapes[i].mesh.indices[index_offset + 2].texcoord_index));

            nTriangles++;
            index_offset += fnum;
        }
    }

    mesh->m_triangle_num = nTriangles;
    mesh->m_cpu_p.copyFrom(vertices.size(), HOST_BUFFER, vertices.data());
    mesh->m_cpu_n.copyFrom(normals.size(), HOST_BUFFER, normals.data());
    mesh->m_cpu_uv.copyFrom(uvs.size(), HOST_BUFFER, uvs.data());
    mesh->m_cpu_index.copyFrom(indices.size(), HOST_BUFFER, indices.data());
    */
    return true;    
}
