#include "spectrum.h"

Float Clamp(Float a) {
    if (a > 1) return 1;
    else if (a < 0) return 0;
    else return a;
}

Float Spectrum::operator[](int idx) const
{
    if (idx == 0) return r;
    else if (idx == 1) return g;
    else return b; 
}

void SpectrumToUnsignedChar(
    const Spectrum& s, 
    unsigned char* const uc, 
    int len)
{
    for (int i = 0; i < 3; i++) {
        uc[i] = (unsigned char)(Clamp(s[i]) * 255);
    }
    if (len == 4) {
        uc[3] = 255;
    }
}

