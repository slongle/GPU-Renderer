#pragma once

#include "utility/types.h"

class MemoryArena {
public:
    MemoryArena(uint8* ptr = nullptr) :m_ptr(ptr), m_size(0) {}

    uint8* byteAlloc(const uint64 size)
    {
        const uint64 base = m_size; 
        m_size += size;
        return m_ptr + base;
    }

    template<typename T>
    T* alloc(const uint64 size)
    {
        return (T*)byteAlloc(size * sizeof(T));
    }

    uint64 size() const
    {
        return m_size;
    }

    uint8* ptr() const
    {
        return m_ptr;
    }

    uint8* m_ptr;
    uint64 m_size;
};