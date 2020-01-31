#pragma once

#include "renderer/fwd.h"

class Camera {
public:    

    Camera()         
    {        
        m_eye = make_float3(0, 0, -1);        
        m_dir = make_float3(0, 0, -1);
        m_up = make_float3(0, 1, 0);
        m_fov = make_float2(degToRad(40));
        m_yaw = 0;
        m_pitch = PI;
        m_aspect_ratio = 1;
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

    float3 m_eye;
    float3 m_dir;
    float3 m_up;
    float2 m_fov;

    float m_yaw, m_pitch;
    float m_aspect_ratio;
};