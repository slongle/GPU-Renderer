#include "pbrtloader.h"
#include "renderer/core/api.h"

#include <functional>
#include <vector>

std::shared_ptr<Renderer> 
Parse(std::unique_ptr<Tokenizer> tokenizer);

std::shared_ptr<Renderer> 
PBRTLoader::Load()
{
    ASSERT(m_filepath.extension() == "pbrt", "The extension of scene is not .pbrt");
    std::unique_ptr<Tokenizer> tokenizer = Tokenizer::CreateFromFile(m_filepath.str());
    return Parse(std::move(tokenizer));
}

std::unique_ptr<Tokenizer> Tokenizer::CreateFromFile(const std::string filename)
{
    FILE* f = fopen(filename.c_str(), "r");
    ASSERT(f, "Can't open file " + filename);

    std::string str;
    int ch;
    while ((ch = fgetc(f)) != EOF) {
        str.push_back(char(ch));
    }

    return std::unique_ptr<Tokenizer>(new Tokenizer(std::move(str)));
}


std::string_view Tokenizer::Next()
{
    while (true) {
        const char* startPos = m_pos;
        int ch = GetChar();
        if (ch == EOF) {
            return std::string_view(startPos, 0);
        }
        else if (ch == '#') {
            while ((ch = GetChar()) != EOF) {
                if (ch == '\n') {

                    break;
                }
            }
            return std::string_view(startPos, m_pos - startPos);
        }
        else if (ch == '"') {
            while ((ch = GetChar()) != '"') {}
            return std::string_view(startPos, m_pos - startPos);
        }
        else if (ch == '[' || ch == ']') {
            return std::string_view(startPos, m_pos - startPos);
        }
        else if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {            

        }
        else {
            while ((ch = GetChar()) != EOF) {
                if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || 
                    ch == '"' || ch == '[' || ch == ']') {                    
                    UngetChar();
                    break;
                }
            }
            return std::string_view(startPos, m_pos - startPos);
        }
    }
}

Tokenizer::Tokenizer(std::string str)
    :m_content(std::move(str))
{
    m_pos = m_content.data();
    m_end = m_pos + m_content.size();
}

int Tokenizer::GetChar()
{
    if (m_pos == m_end) {
        return EOF;
    }
    int ch = *m_pos++;
    if (ch == '\n') {
        m_loc.m_line++;
        m_loc.m_column = 0;
    } else {
        m_loc.m_column++;
    }
    return ch;
}

void Tokenizer::UngetChar()
{
    m_pos--;
    if (*m_pos == '\n') {
        m_loc.m_line--;
    }
}

struct ParameterItem {
    std::string m_name;
    std::vector<std::string> m_val;
};

bool isQuotedString(const std::string_view& str) {
    return str.front() == '"' && str.back() == '"';
}

void dequotedString(std::string_view& str) {
    str.remove_prefix(1);
    str.remove_suffix(1);
}

enum {
    TYPE_BOOL,
    TYPE_INTEGER,
    TYPE_FLOAT,
    TYPE_STRING,
    TYPE_VECTOR,
    TYPE_POINT,
    TYPE_NORMAL,
    TYPE_RGB,
};


void lookUpTypeAndName(const std::string& str,std::string& name, int& type) {
    auto skipSpace = [&](std::string::const_iterator iter) {
        while (iter != str.end() && (*iter == ' ' || *iter == '\t')) ++iter;
        return iter;
    };

    auto skipToSpace = [&](std::string::const_iterator iter) {
        while (iter != str.end() && (*iter != ' ' && *iter != '\t')) ++iter;
        return iter;
    };

    std::string::const_iterator typeBegin = skipSpace(str.begin());
    std::string::const_iterator typeEnd = skipToSpace(typeBegin);
    std::string_view typeStr(&(*typeBegin), typeEnd - typeBegin);    
    if (typeStr == "bool") {
        type = TYPE_BOOL;
    }
    else if (typeStr == "integer") {
        type = TYPE_INTEGER;
    }
    else if (typeStr == "float") {
        type = TYPE_FLOAT;
    }
    else if (typeStr == "string") {
        type = TYPE_STRING;
    } 
    else if (typeStr == "rgb") {
        type = TYPE_RGB;
    }
    else if (typeStr == "point") {
        type = TYPE_POINT;
    } 
    else if (typeStr == "normal") {
        type = TYPE_NORMAL;
    }
    else {
        ASSERT(0, "Can't support type " + std::string(typeStr));
    }

    std::string::const_iterator nameBegin = skipSpace(typeEnd);
    std::string::const_iterator nameEnd = skipToSpace(nameBegin);
    name = std::string(nameBegin, nameEnd);
}

void AddParameters(ParameterSet& params, const ParameterItem& item) {
    std::string attributeName;
    int attributeType;
    lookUpTypeAndName(item.m_name, attributeName, attributeType);
    if (attributeType == TYPE_INTEGER) {
        std::vector<int> attributeVal;
        char* endPtr;
        for (const std::string& v : item.m_val) {
            attributeVal.push_back(strtol(v.c_str(), &endPtr, 10));
        }
        params.AddInt(attributeName, std::move(attributeVal));
    }
    else if (attributeType == TYPE_FLOAT) {
        std::vector<Float> attributeVal;
        char* endPtr;
        for (const std::string& v : item.m_val) {
            attributeVal.push_back(strtof(v.c_str(), &endPtr));
        }
        params.AddFloat(attributeName, std::move(attributeVal));
    }
    else if (attributeType == TYPE_STRING) {
        params.AddString(attributeName, std::move(item.m_val));
    }
    else if (attributeType == TYPE_POINT) {
        std::vector<Point3f> attributeVal;
        const std::vector<std::string>& val = item.m_val;
        ASSERT(val.size() % 3 == 0, "The number of value is not a multiple of 3");
        char* endPtr;
        for (int i = 0; i < val.size(); i += 3) {            
            attributeVal.push_back(Point3f(strtof(val[i].c_str(), &endPtr),
                strtof(val[i + 1].c_str(), &endPtr), strtof(val[i + 2].c_str(), &endPtr)));
        }
        params.AddPoint(attributeName, std::move(attributeVal));
    }
    else if (attributeType == TYPE_NORMAL) {
        std::vector<Normal3f> attributeVal;
        const std::vector<std::string>& val = item.m_val;
        ASSERT(val.size() % 3 == 0, "The number of value is not a multiple of 3");
        char* endPtr;
        for (int i = 0; i < val.size(); i += 3) {
            attributeVal.push_back(Normal3f(strtof(val[i].c_str(), &endPtr),
                strtof(val[i + 1].c_str(), &endPtr), strtof(val[i + 2].c_str(), &endPtr)));
        }
        params.AddNormal(attributeName, std::move(attributeVal));
    }
    else if (attributeType == TYPE_RGB) {
        std::vector<Float> attributeVal;
        const std::vector<std::string>& val = item.m_val;
        ASSERT(val.size() % 3 == 0, "The number of value is not a multiple of 3");
        char* endPtr;
        for (int i = 0; i < val.size(); i ++) {
            attributeVal.push_back(strtof(val[i].c_str(), &endPtr));
        }
        params.AddSpectrum(attributeName, std::move(attributeVal));
    }
    else {
        ASSERT(0, "Can't support type " + std::to_string(attributeType));
    }
}

std::shared_ptr<Renderer>
Parse(std::unique_ptr<Tokenizer> tokenizer) {
    std::shared_ptr<Renderer> renderer;

    bool ungetTokenSet = false;
    std::string_view ungetTokenValue;

    std::function<std::string_view()>
        nextToken = [&]() -> std::string_view {
        std::string_view token = tokenizer->Next();
        if (token.empty()) {
            return token;
        }
        if (token[0] == '#') {
            return nextToken();
        }
        return token;
    };

    std::function<void(const std::string_view&)>
        ungetToken = [&](const std::string_view& token) {
        ungetTokenSet = true;
        ungetTokenValue = token;
    };

    std::function<ParameterSet()>
        parseParameters = [&]() {

        auto addVal = [&](ParameterItem& item, std::string_view& token) {
            if (isQuotedString(token)) {
                dequotedString(token);
                item.m_val.emplace_back(token);
            }
            else {
                item.m_val.emplace_back(token);
            }
        };

        ParameterSet params;
        while (true) {
            std::string_view token = nextToken();
            ParameterItem item;
            if (!isQuotedString(token)) {
                ungetToken(token);
                break;
            }
            else {
                dequotedString(token);
                item.m_name = std::string(token);
                token = nextToken();
                ASSERT(token[0] == '[', "Expected '\"'");
                token = nextToken();
                while (token[0] != ']') {
                    addVal(item, token);
                    token = nextToken();
                }
            }
            AddParameters(params, item);
        }
        return params;
    };

    auto parseParameterList = [&](
        std::function<void(const std::string&, ParameterSet)> apiFunc) {
        std::string_view token = nextToken();
        ASSERT(isQuotedString(token), "Expected quoted string");
        dequotedString(token);
        std::string type(token);        

        ParameterSet params = parseParameters();

        apiFunc(type, params);
    };

    while (true) {
        std::string_view token;
        if (ungetTokenSet) {
            ungetTokenSet = false;
            token = ungetTokenValue;
        }
        else {
            token = nextToken();
        }
        if (token.empty()) {
            break;
        }
        switch (token[0]) {
        case 'A':
            if (token == "AttributeBegin") {
                apiAttributeBegin();
            }
            else if (token == "AttributeEnd") {
                apiAttributeEnd();
            }
            else if (token == "AreaLightSource") {
                parseParameterList(apiAreaLightSource);
            }
            break;
        case 'C' :
            if (token == "Camera") {
                parseParameterList(apiCamera);
            }
            break;
        case 'F' :
            if (token == "Film") {
                parseParameterList(apiFilm);
            }
            break;
        case 'I' :
            if (token == "Integrator") {
                parseParameterList(apiIntegrator);
            }
            break;
        case 'M':
            if (token == "MakeNamedMaterial") {
                parseParameterList(apiMakeNamedMaterial);
            } 
            break;
        case 'N':
            if (token == "NamedMaterial") {
                parseParameterList(apiNamedMaterial);
            }
            break;
        case 'P' :
            if (token == "PixelFilter") {
                parseParameterList(apiFilter);
            }
            break;
        case 'S' :
            if (token == "Sampler") {
                parseParameterList(apiSampler);
            }
            else if (token == "Shape") {
                parseParameterList(apiShape);
            }
            break;
        case 'T' :
            if (token == "Transform") {
                token = nextToken();
                char* endPtr;
                Float m[16];
                for (int i = 0; i < 16; i++) {
                    token = nextToken();
                    m[i] = strtof(token.data(), &endPtr);
                }
                apiTransform(m);
            }
            else if (token == "TransformBegin") {
                apiTransformBegin();
            } 
            else if (token == "TransformEnd") {
                apiTransformEnd();
            }
            break;
        case 'W':
            if (token == "WorldBegin") {
                apiWorldBegin();
            }
            else if (token == "WorldEnd") {
                renderer = apiWorldEnd();
            }
            break;
        }        
    }
    return renderer;
}