#pragma once

#include <cooperative_groups.h>
using namespace cooperative_groups;

__device__ __forceinline__
__device__ unsigned int warp_increment(unsigned int* ctr) {
    auto g = coalesced_threads();
    int warp_res;
    if (g.thread_rank() == 0)
        warp_res = atomicAdd(ctr, g.size());
    return g.shfl(warp_res, 0) + g.thread_rank();
}
