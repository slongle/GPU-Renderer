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

    int subdiv = 10;
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

    /*for (int i = 0; i < p.size(); i++) {
        std::cout << "v " << p[i].x << ' ' << p[i].y << ' ' << p[i].z << std::endl;
    }

    for (int i = 0; i < indices.size(); i+=3) {
        std::cout << "f " << indices[i] << ' ' << indices[i+1] << ' ' << indices[i+2] << std::endl;
    }*/


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
