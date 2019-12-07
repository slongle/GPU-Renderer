#include "mitsubaloader.h"

#include "ext/pugixml/pugixml.hpp"

#include <fstream>

std::string MitsubaLoader::getOffset(ptrdiff_t pos) const
{    
    std::fstream is(m_filepath.str());
    char buffer[1024];
    int line = 0, linestart = 0, offset = 0;
    while (is.good()) {
        is.read(buffer, sizeof(buffer));
        for (int i = 0; i < is.gcount(); ++i) {
            if (buffer[i] == '\n') {
                if (offset + i >= pos)
                    return tfm::format("line %i, col %i", line + 1, pos - linestart);
                ++line;
                linestart = offset + i;
            }
        }
        offset += (int)is.gcount();
    }
    return "byte offset " + std::to_string(pos);
}


void MitsubaLoader::load()
{
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(m_filepath.str().c_str());

    ASSERT(result, "Error while parsing \"" + m_filepath.filename() + "\": " + result.description() + \
                   " (at " + getOffset(result.offset) + ")");

}

