#pragma once
#ifndef __MEMORY_H
#define __MEMORY_H

#include <memory>
#include <list>
#include <algorithm>
#include <cstddef>

#define HAVE_ALIGNED_MALLOC
#define L1_CACHE_LINE_SIZE 64

#define ALLOCA(TYPE, COUNT) (TYPE*)alloca((COUNT) * sizeof(TYPE))
#define ARENA_ALLOCA(ARENA, TYPE) new ((ARENA).Alloc(sizeof(TYPE))) TYPE

inline 
void* AllocAligned(const size_t size) {
#if defined(HAVE_ALIGNED_MALLOC)
    return _aligned_malloc(size, L1_CACHE_LINE_SIZE);
#elif defined(HAVE_POSIX_MEMALIGN)
    void* ptr;
    if (posix_memalign(&ptr, L1_CACHE_LINE_SIZE, size) != 0) ptr = nullptr;
    return ptr;
#else
    return memalign(L1_CACHE_LINE_SIZE, size);
#endif
}

template<typename T>
inline
T* AllocAligned(const size_t count) {
    return (T*)AllocAligned(count * sizeof(T));
}

inline
void FreeAligned(void* ptr) {
    if (!ptr) return;
#if defined(HAVE_ALIGNED_MALLOC)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

class MemoryArena {
public:
    MemoryArena(size_t m_blockSize = 1 << 18) :blockSize(m_blockSize) {}
    ~MemoryArena() 
    {
        FreeAligned(currentBlock);
        for (auto& block : usedBlocks) FreeAligned(block.second);
        for (auto& block : availableBlocks) FreeAligned(block.second);
    }

    void* Alloc(size_t nBytes) 
    {        
        size_t align = 16;
        nBytes = (nBytes + (align - 1)) & (~(align - 1));

        if (currentBlockPos + nBytes > currentAllocSize) {
            if (currentBlock) {
                usedBlocks.emplace_back(currentAllocSize, currentBlock);
                currentBlock = nullptr;
            }

            for (auto& availableBlock : availableBlocks) {
                if (availableBlock.first > nBytes) {
                    currentAllocSize = availableBlock.first;
                    currentBlockPos = 0;
                    currentBlock = availableBlock.second;
                    break;
                }
            }

            if (!currentBlock) {
                currentAllocSize = max(blockSize, nBytes);
                currentBlockPos = 0;
                currentBlock = AllocAligned<uint8_t>(currentAllocSize);
            }
        }

        void* ret = currentBlock + currentBlockPos;
        currentBlockPos += nBytes;
        return ret;
    }
    
    template<typename T>
    T* Alloc(size_t n = 1, bool runConstructor = true) 
    {
        T* ret = (T*)Alloc(sizeof(T) * n);
        if (runConstructor) {
            for (size_t i = 0; i < n; i++)
                new (&ret[i]) T();
        }
        return ret;
    }

    void Reset() 
    {
        currentBlockPos = currentAllocSize = 0;
        currentBlock = nullptr;
        availableBlocks.splice(availableBlocks.begin(), usedBlocks);
    }

private:
    const size_t blockSize;
    size_t currentBlockPos = 0, currentAllocSize = 0;
    uint8_t* currentBlock = nullptr;
    std::list<std::pair<size_t, uint8_t*>> usedBlocks, availableBlocks;
};

#endif // __MEMORY_H