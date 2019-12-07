#pragma once
#ifndef __HELPER_LOGGER_H
#define __HELPER_LOGGER_H

#include <iostream>

#define ASSERT(CONDITION, DESCRIPTION) \
    do { \
        if (!(CONDITION)) { \
            std::cerr<<"\nAssertion : " << (DESCRIPTION) << "\nFile : "<<__FILE__<<"\nFunction : " \
                     <<__FUNCTION__<<"\nLine : "<<__LINE__ << std::endl, exit(-1); \
        } \
    } while (0)            

#endif // !__HELPER_LOGGER_H
