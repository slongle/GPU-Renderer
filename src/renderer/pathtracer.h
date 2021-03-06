#pragma once

#include "renderer/scene.h"
#include "renderer/camera.h"
#include "renderer/framebuffer.h"

struct PTOptions
{
    PTOptions(const uint32 max_path_length = 33)
        :m_max_path_length(max_path_length)
    {}

    uint32 m_max_path_length;
};

class PathTracer {
public:
    PathTracer(const std::string& filename);

    ~PathTracer();

    void init();
    void initScene();
    void initQueue();
    void render(uint32* output = nullptr);
    void render(uint32 num);
    void output(const std::string& filename);

    void zoom(float d);
    void translate(float x, float y);
    void rotate(float yaw, float pitch);

    void reset();
    void resize(uint32 width, uint32 height);

    int2 getResolution() const;

private:
    PTOptions m_options;
    Scene m_scene;
    Buffer<DEVICE_BUFFER, uint8> m_memory_pool;

    bool m_reset;
    uint32 m_sample_num;
    uint64 m_sum_bounce;
};