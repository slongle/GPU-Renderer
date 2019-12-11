#include "random.h"

void RandomSampler::Init(unsigned int v0, unsigned int v1, unsigned int backoff)
{
    unsigned int s0 = 0;
    for (unsigned int n = 0; n < backoff; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    m_seed = v0;
}

Float RandomSampler::Next()
{
    m_seed = (1664525u * m_seed + 1013904223u);
    return float(m_seed & 0x00FFFFFF) / float(0x01000000);
}
