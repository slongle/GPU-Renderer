#pragma once

#include "renderer/fwd.h"
#include "renderer/spectrum.h"

class FrameBuffer;

struct FrameBufferView
{
    FrameBufferView(const FrameBuffer* fb);

    HOST_DEVICE
    void addRadiance(uint32 idx, Spectrum col)
    {
        m_buffer[idx] += col;
    }

    HOST_DEVICE
        void addSampleNum(uint32 idx)
    {
        m_sample_num[idx] ++;
    }

    HOST_DEVICE
    uint32 getIdx(const uint2 pixel) const { return pixel.x + pixel.y * m_resolution_x; }



    uint32 m_resolution_x;
    uint32 m_resolution_y;
    Spectrum* m_buffer;
    uint32* m_sample_num;
};

class FrameBuffer 
{
public:
    FrameBuffer(const uint32_t res_x = 0, const uint32_t res_y = 0) :
        m_resolution_x(res_x),
        m_resolution_y(res_y),
        m_buffer(res_x * res_y),
        m_sample_num(res_x* res_y)
    {
    }

    void output(const std::string& filename);

    void resize(const uint32_t res_x, const uint32_t res_y) 
    {
        m_resolution_x = res_x;
        m_resolution_y = res_y;
        m_buffer.resize(size());
        m_sample_num.resize(size());
        clear();
    }

    void clear()
    {
        m_sample_num.clear();
        m_buffer.clear();
    }

    FrameBufferView view() const { return FrameBufferView(this); }
    size_t size() const { return m_resolution_x * m_resolution_y; }

    uint32 m_resolution_x;
    uint32 m_resolution_y;
    Buffer<DEVICE_BUFFER, Spectrum> m_buffer;
    Buffer<DEVICE_BUFFER, uint32> m_sample_num;
};