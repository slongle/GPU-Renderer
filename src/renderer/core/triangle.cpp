#include "ext/rply/rply.h"

#include "triangle.h"

TriangleMesh::TriangleMesh(
    Transform objToWorld,
    std::vector<int>& indices,
    std::vector<Point3f>& p,
    std::vector<Normal3f>& n,
    std::vector<float>& uv)
    : m_triangleNum(indices.size() / 3), m_vertexNum(p.size())
{
    m_indices = new int[m_triangleNum * 3];
    for (int i = 0; i < m_triangleNum * 3; i++) {
        m_indices[i] = indices[i];
    }
    m_P = new Point3f[m_vertexNum];
    for (int i = 0; i < m_vertexNum; i++) {
        m_P[i] = objToWorld(p[i]);
    }
    if (n.size() != 0) {
        m_N = new Normal3f[m_vertexNum];
        for (int i = 0; i < m_vertexNum; i++) {
            m_N[i] = objToWorld(n[i]);
        }
    }
    if (uv.size() != 0) {
        m_UV = new Point2f[m_vertexNum];
        for (int i = 0; i < m_vertexNum; i += 2) {
            m_UV[i] = Point2f(uv[i], uv[i + 1]);
        }
    }
}


Triangle::Triangle(
    TriangleMesh* triangleMeshPtr, int index)
    : m_triangleMeshPtr(triangleMeshPtr), m_index(index), m_triangleMeshID(-1) {}


TriangleMesh*
ConvertSphereToTriangleMesh(
    Float radius,
    Transform& objToWorld)
{
    std::vector<int> indices;
    std::vector<Point3f> p;
    std::vector<Normal3f> n;
    std::vector<Float> uv;

    int subdiv = 8;
    int nLatitude = subdiv;
    int nLongitude = 2 * subdiv;
    Float latitudeDeltaAngle = Pi / nLatitude;
    Float longitudeDeltaAngle = 2 * Pi / nLongitude;

    p.emplace_back(0, 0, radius);
    n.emplace_back(0, 0, 1);
    for (int i = 1; i < nLatitude; i++) {
        Float theta = i * latitudeDeltaAngle;
        for (int j = 0; j < nLongitude; j++) {
            Float phi = j * longitudeDeltaAngle;
            Float x = sin(theta) * cos(phi);
            Float y = sin(theta) * sin(phi);
            Float z = cos(theta);
            p.emplace_back(x * radius, y * radius, z * radius);
            n.emplace_back(x, y, z);
        }
    }
    p.emplace_back(0, 0, -radius);
    n.emplace_back(0, 0, -1);

    for (int i = 0; i < nLongitude; i++) {
        int a = i + 1, b = a + 1;
        if (i == nLongitude - 1) b = 1;
        indices.push_back(a);
        indices.push_back(b);
        indices.push_back(0);
    }

    for (int i = 2; i < nLatitude; i++) {
        for (int j = 0; j < nLongitude; j++) {
            int a = (i - 1) * nLongitude + j + 1, b = a + 1, c = a - nLongitude, d = c + 1;
            if (j == nLongitude - 1) b = (i - 1) * nLongitude + 1, d = b - nLongitude;
            indices.push_back(a);
            indices.push_back(b);
            indices.push_back(c);

            indices.push_back(c);
            indices.push_back(b);
            indices.push_back(d);
        }
    }

    int bottomIdx = nLongitude * (nLatitude - 1) + 1;
    for (int i = 0; i < nLongitude; i++) {
        int a = (nLatitude - 2) * nLongitude + i + 1, b = a + 1;
        if (i == nLongitude - 1) b = (nLatitude - 2) * nLongitude + 1;
        indices.push_back(a);
        indices.push_back(bottomIdx);
        indices.push_back(b);
    }

    return new TriangleMesh(objToWorld, indices, p, n, uv);
}

std::vector<std::shared_ptr<Triangle>>
CreateTriangleMeshShape(
    const ParameterSet& params,
    Transform objToWorld,
    Transform worldToObj)
{
    std::vector<int> indices = params.GetInts("indices");
    std::vector<Point3f> p = params.GetPoints("P");
    std::vector<Normal3f> n = params.GetNormals("N");
    std::vector<Float> uv = params.GetFloats("uv", std::vector<Float>());

    TriangleMesh* triangleMesh = new TriangleMesh(objToWorld, indices, p, n, uv);
    std::vector<std::shared_ptr<Triangle>> triangles;
    int triangleNum = indices.size() / 3;
    for (int i = 0; i < triangleNum; i++) {
        triangles.push_back(std::make_shared<Triangle>(triangleMesh, i));
    }

    return triangles;
}

std::vector<std::shared_ptr<Triangle>> 
CreateSphereShape(
    const ParameterSet& params, 
    Transform objToWorld, 
    Transform worldToObj)
{
    Float radius = params.GetFloat("radius");
    TriangleMesh* mesh = ConvertSphereToTriangleMesh(radius, objToWorld);    

    std::vector<std::shared_ptr<Triangle>> triangles;
    int triangleNum = mesh->m_triangleNum;
    for (int i = 0; i < triangleNum; i++) {
        triangles.push_back(std::make_shared<Triangle>(mesh, i));
    }

    return triangles;
}

struct CallbackContext {
    Point3f* p;
    Normal3f* n;
    Point2f* uv;
    int* indices;
    int* faceIndices;
    int indexCtr, faceIndexCtr;
    int face[4];
    bool error;
    int vertexCount;

    CallbackContext()
        : p(nullptr),
        n(nullptr),
        uv(nullptr),
        indices(nullptr),
        faceIndices(nullptr),
        indexCtr(0),
        faceIndexCtr(0),
        error(false),
        vertexCount(0) {}

    ~CallbackContext() {
        delete[] p;
        delete[] n;
        delete[] uv;
        delete[] indices;
        delete[] faceIndices;
    }
};

void rply_message_callback(p_ply ply, const char* message) {    
}

/* Callback to handle vertex data from RPly */
int rply_vertex_callback(p_ply_argument argument) {
    Float** buffers;
    long index, flags;

    ply_get_argument_user_data(argument, (void**)&buffers, &flags);
    ply_get_argument_element(argument, nullptr, &index);

    int bufferIndex = (flags & 0xF00) >> 8;
    int stride = (flags & 0x0F0) >> 4;
    int offset = flags & 0x00F;

    Float* buffer = buffers[bufferIndex];
    if (buffer)
        buffer[index * stride + offset] =
        (float)ply_get_argument_value(argument);

    return 1;
}

/* Callback to handle face data from RPly */
int rply_face_callback(p_ply_argument argument) {
    CallbackContext* context;
    long flags;
    ply_get_argument_user_data(argument, (void**)&context, &flags);

    if (flags == 0) {
        // Vertex indices

        long length, value_index;
        ply_get_argument_property(argument, nullptr, &length, &value_index);

        if (length != 3 && length != 4) {           
            return 1;
        }
        else if (value_index < 0) {
            return 1;
        }
        if (length == 4) {
            ASSERT(0, "Can't support quad");
        }

        if (value_index >= 0) {
            int value = (int)ply_get_argument_value(argument);
            if (value < 0 || value >= context->vertexCount) {
                context->error = true;
            }
            context->face[value_index] = value;
        }

        if (value_index == length - 1) {
            for (int i = 0; i < 3; ++i)
                context->indices[context->indexCtr++] = context->face[i];

            if (length == 4) {
                /* This was a quad */
                context->indices[context->indexCtr++] = context->face[3];
                context->indices[context->indexCtr++] = context->face[0];
                context->indices[context->indexCtr++] = context->face[2];
            }
        }
    }
    else {
        // Face indices
        context->faceIndices[context->faceIndexCtr++] =
            (int)ply_get_argument_value(argument);
    }

    return 1;
}

std::vector<std::shared_ptr<Triangle>> CreatePLYMeshShape(
    const ParameterSet& params,
    const Transform& o2w, 
    const Transform& w2o) 
{    
    const std::string filename = params.GetString("filename", "");
    filesystem::path path = getFileResolver()->resolve(filename);
    p_ply ply = ply_open(path.str().c_str(), rply_message_callback, 0, nullptr);
    if (!ply) {        
        return std::vector<std::shared_ptr<Triangle>>();
    }

    if (!ply_read_header(ply)) {        
        return std::vector<std::shared_ptr<Triangle>>();
    }

    p_ply_element element = nullptr;
    long vertexCount = 0, faceCount = 0;

    /* Inspect the structure of the PLY file */
    while ((element = ply_get_next_element(ply, element)) != nullptr) {
        const char* name;
        long nInstances;

        ply_get_element_info(element, &name, &nInstances);
        if (!strcmp(name, "vertex"))
            vertexCount = nInstances;
        else if (!strcmp(name, "face"))
            faceCount = nInstances;
    }

    if (vertexCount == 0 || faceCount == 0) {
        return std::vector<std::shared_ptr<Triangle>>();
    }

    CallbackContext context;

    if (ply_set_read_cb(ply, "vertex", "x", rply_vertex_callback, &context,
        0x030) &&
        ply_set_read_cb(ply, "vertex", "y", rply_vertex_callback, &context,
            0x031) &&
        ply_set_read_cb(ply, "vertex", "z", rply_vertex_callback, &context,
            0x032)) {
        context.p = new Point3f[vertexCount];
    }
    else {
        return std::vector<std::shared_ptr<Triangle>>();
    }

    if (ply_set_read_cb(ply, "vertex", "nx", rply_vertex_callback, &context,
        0x130) &&
        ply_set_read_cb(ply, "vertex", "ny", rply_vertex_callback, &context,
            0x131) &&
        ply_set_read_cb(ply, "vertex", "nz", rply_vertex_callback, &context,
            0x132))
        context.n = new Normal3f[vertexCount];

    /* There seem to be lots of different conventions regarding UV coordinate
     * names */
    if ((ply_set_read_cb(ply, "vertex", "u", rply_vertex_callback, &context,
        0x220) &&
        ply_set_read_cb(ply, "vertex", "v", rply_vertex_callback, &context,
            0x221)) ||
            (ply_set_read_cb(ply, "vertex", "s", rply_vertex_callback, &context,
                0x220) &&
                ply_set_read_cb(ply, "vertex", "t", rply_vertex_callback, &context,
                    0x221)) ||
                    (ply_set_read_cb(ply, "vertex", "texture_u", rply_vertex_callback,
                        &context, 0x220) &&
                        ply_set_read_cb(ply, "vertex", "texture_v", rply_vertex_callback,
                            &context, 0x221)) ||
                            (ply_set_read_cb(ply, "vertex", "texture_s", rply_vertex_callback,
                                &context, 0x220) &&
                                ply_set_read_cb(ply, "vertex", "texture_t", rply_vertex_callback,
                                    &context, 0x221)))
        context.uv = new Point2f[vertexCount];

    /* Allocate enough space in case all faces are quads */
    context.indices = new int[faceCount * 6];
    context.vertexCount = vertexCount;

    ply_set_read_cb(ply, "face", "vertex_indices", rply_face_callback, &context,
        0);
    if (ply_set_read_cb(ply, "face", "face_indices", rply_face_callback, &context,
        1))
        // Extra space in case they're quads
        context.faceIndices = new int[faceCount];

    if (!ply_read(ply)) {
        ply_close(ply);
        return std::vector<std::shared_ptr<Triangle>>();
    }

    ply_close(ply);

    if (context.error) return std::vector<std::shared_ptr<Triangle>>();

    ParameterSet pa;
    std::vector<int> indices;
    std::vector<Point3f> p;
    std::vector<Normal3f> n;
    std::vector<Float> uv;
    for (int i = 0; i < context.indexCtr; i++) indices.push_back(context.indices[i]);
    for (int i = 0; i < context.vertexCount; i++) p.push_back(context.p[i]);
    if (context.n != nullptr) {
        for (int i = 0; i < context.vertexCount; i++) n.push_back(context.n[i]);
    }
    if (context.uv != nullptr) {
        for (int i = 0; i < context.vertexCount; i++) {
            uv.push_back(context.uv[i].x);
            uv.push_back(context.uv[i].y);
        }
    }
    pa.AddInt("indices", indices);
    pa.AddPoint("P", p);
    pa.AddNormal("N", n);
    pa.AddFloat("uv", uv);
    return CreateTriangleMeshShape(pa, o2w, w2o);
}
