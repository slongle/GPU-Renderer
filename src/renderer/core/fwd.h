#pragma once
#ifndef __FWD_H
#define __FWD_H

#include "utility/helper_logger.h"

#include "ext/tinyformat/tinyformat.h"
#include "ext/filesystem/resolver.h"


inline filesystem::resolver* getFileResolver() {
    static filesystem::resolver* resolver = new filesystem::resolver();
    return resolver;
}

#endif // !__FWD_H
