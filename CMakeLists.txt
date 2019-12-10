﻿CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT("GPU Renderer")

FIND_PACKAGE(CUDA REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")

# MESSAGE(${CMAKE_CURRENT_SOURCE_DIR})
# MESSAGE(${CMAKE_CURRENT_BINARY_DIR})
# MESSAGE(${CMAKE_BINARY_DIR})
# MESSAGE(${CMAKE_SOURCE_DIR})

INCLUDE_DIRECTORIES(
    "${CMAKE_SOURCE_DIR}/src/"    
)

LINK_DIRECTORIES(
)

CUDA_ADD_EXECUTABLE(GPURenderer
    
)

#Include sub-projects.
#add_subdirectory ("MachineLearning")

#target_link_libraries(MachineLearning)