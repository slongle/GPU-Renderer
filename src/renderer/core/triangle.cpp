#include "triangle.h"

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