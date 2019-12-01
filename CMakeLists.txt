cmake_minimum_required (VERSION 3.8)

project ("GPU-Renderer")


include_directories(
    # MachineLearning include files
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(GPURenderer
    # Header files
    # include/MachineLearning/*.h
    
    # Source code files
    src/main.cpp
)

# Include sub-projects.
# add_subdirectory ("MachineLearning")

# target_link_libraries(MachineLearning )