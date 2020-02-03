#include "imageio.h"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb_image/stb_image.h"
#endif // !STB_IMAGE_IMPLEMENTATION

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ext/stb_image/stb_image_write.h"
#endif // !STB_IMAGE_WRITE_IMPLEMENTATION

void ReadImage(
    const std::string& filename,
    int* width, 
    int* height,
    uint8*& buffer)
{
    int nchannels;
    buffer = stbi_load(filename.c_str(), width, height, &nchannels, 3);    
}

void ReadImage(
    const std::string& filename,
    int* width,
    int* height,
    float*& buffer)
{
    int nchannels;
    buffer = stbi_loadf(filename.c_str(), width, height, &nchannels, 3);
}

void ReadImage(
    const std::string& filename,
    int* width,
    int* height,
    std::vector<uint8>& buffer)
{
    int nchannels;
    uint8* ptr= stbi_load(filename.c_str(), width, height, &nchannels, 3);
    buffer.assign(ptr, ptr + (*width) * (*height) * 3);
    stbi_image_free(ptr);
}

void ReadImage(
    const std::string& filename,
    int* width,
    int* height,
    std::vector<float>& buffer)
{
    int nchannels;
    float* ptr = stbi_loadf(filename.c_str(), width, height, &nchannels, 3);
    buffer.assign(ptr, ptr + (*width) * (*height) * 3);
    stbi_image_free(ptr);
}

void WriteImage(
    const std::string& filename,
    const int& width,
    const int& height,
    uint8* buffer)
{

    //stbi_flip_vertically_on_write(true);

    std::string ext = getFileExtension(filename);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".png")
    {
        stbi_write_png(filename.c_str(), width, height, 3, buffer, 0);
    }
    else if (ext == ".hdr")
    {
        stbi_write_hdr(filename.c_str(), width, height, 3, (float*)buffer);
    }
    else
    {
        assert(false);
    }

}
