#pragma once

#include "renderer/fwd.h"
#include "renderer/transform.h"
#include "renderer/ray.h"

class Camera {
public:    

    Camera()         
    {        
        m_eye = make_float3(0, 2, 7);        
        m_dir = make_float3(0, -0.2, -1);
        m_up = make_float3(0, 1, 0);
        m_fov = make_float2(degToRad(40));
        m_yaw = atan2(m_dir.x, m_dir.z);
        m_pitch = asin(m_dir.y);
        m_aspect_ratio = 1;
    }

    void setup(const Transform& c2w, const float& fov, const uint32& width, const uint32& height, const bool& free)
    {
        m_cameraToWorld = c2w;
        m_fov = make_float2(degToRad(fov));
        m_resolution_x = width;
        m_resolution_y = height;
        m_aspect_ratio = float(m_resolution_x) / m_resolution_y;
        m_free = free;
        //m_free = true;
        if (!m_free)
        {
            Transform CameraToRaster =
                Scale(m_resolution_x, m_resolution_y, 1) *
                Scale(-0.5, -0.5 * m_aspect_ratio, 1) *
                Translate(-1, -1. / m_aspect_ratio, 0) *
                Perspective(fov, 1e-4f, 10000.f);
            m_rasterToCamera = Inverse(CameraToRaster);
        }
    }

    void zoom(float d)
    {
        m_eye += m_dir * d;
    }

    void translate(float x, float y)
    {
        float3 dir = normalize(m_dir);
        float3 horizontal_axis = cross(dir, m_up);
        horizontal_axis = normalize(horizontal_axis);
        float3 vertical_axis = cross(horizontal_axis, dir);
        vertical_axis = normalize(vertical_axis);

        m_eye += horizontal_axis * x + vertical_axis * y;
    }

    void rotate(float yaw, float pitch)
    {
        m_yaw += yaw;
        m_pitch += pitch;
        m_dir = make_float3(sin(m_yaw) * cos(m_pitch), sin(m_pitch), cos(m_yaw) * cos(m_pitch));
    }

    void updateAspectRation(float aspect)
    {
        m_aspect_ratio = aspect;
        m_fov.y = atan(tan(m_fov.x * 0.5) / aspect) * 2.0;
    }

    HOST_DEVICE
    Ray generateRay(const float& p_x, const float& p_y) const
    {
        Ray ray;
        if (m_free)
        {
            float3 dir = normalize(m_dir);
            float3 horizontal_axis = cross(dir, m_up);
            horizontal_axis = normalize(horizontal_axis);
            float3 vertical_axis = cross(horizontal_axis, dir);
            vertical_axis = normalize(vertical_axis);

            float3 middle = m_eye + dir;
            float3 horizontal = horizontal_axis * tan(m_fov.x * 0.5f);
            float3 vertical = vertical_axis * tan(m_fov.y * 0.5f);

            float3 point_on_film = middle + (2.f * (p_x / m_resolution_x) - 1.f) * horizontal +
                (2.f * (p_y / m_resolution_y) - 1.f) * vertical;
            
            ray.o = m_eye;
            ray.d = normalize(point_on_film - m_eye);
            ray.tMin = 0.f;
            ray.tMax = 1e34f;
        }
        else
        {
            float3 pCamera = m_rasterToCamera.transformPoint(make_float3(p_x, p_y, 0));            
            ray.o = m_cameraToWorld.transformPoint(make_float3(0.f));            
            ray.d = normalize(m_cameraToWorld.transformVector(normalize(pCamera)));
            ray.tMin = 0.f;
            ray.tMax = 1e34f;
        }

        return ray;
    }

    float3 m_eye;
    float3 m_dir;
    float3 m_up;
    float m_yaw, m_pitch;

    float2 m_fov;
    uint32 m_resolution_x, m_resolution_y;
    float m_aspect_ratio;    

    Transform m_rasterToCamera, m_cameraToWorld;
    bool m_free;
};