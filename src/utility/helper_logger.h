#pragma once
#ifndef __HELPER_LOGGER_H
#define __HELPER_LOGGER_H

#include <iostream>

#define ASSERT(CONDITION, DESCRIPTION) \
    do { \
        if (!(CONDITION)) { \
            exit(-1); \
        } \
    } while (0)            

/*
printf("\nAssertion : %s\nFile : \nFunction : \nLine : \n",\
                   (DESCRIPTION),__FILE__,__FUNCTION__,__LINE__);\
*/
#endif // !__HELPER_LOGGER_H
