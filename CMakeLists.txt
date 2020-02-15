cmake_minimum_required(VERSION 3.1)

project(GPU-Renderer LANGUAGES CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++17 for host code
set(CMAKE_CXX_STANDARD 17)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

########################################
# CUDA Setup
find_package(CUDA 10 REQUIRED)
include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)

list(APPEND CUDA_NVCC_FLAGS ${CUDA_GENERATE_CODE})
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CUDA_VERBOSE_BUILD ON)

if(WIN32)
    # Set up include and lib paths
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
endif(WIN32)
########################################
# OpenGL Setup

########################################
# Set Source Files

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(sources
    src/ext/filesystem/fwd.h
    src/ext/filesystem/path.h
    src/ext/filesystem/resolver.h
    src/ext/pugixml/pugiconfig.hpp
    src/ext/pugixml/pugixml.hpp
    src/ext/pugixml/pugixml.cpp
	src/ext/stb_image/stb_image.h
	src/ext/stb_image/stb_image_write.h
    src/ext/tinyformat/tinyformat.h
	src/ext/rply/rply.h
	src/ext/rply/rply.cpp

    src/GUI/gui.cpp
    src/GUI/gui.h    

	src/utility/buffer.h
    src/utility/exception.h
    src/utility/helper_cuda.h
    src/utility/helper_functions.h
    src/utility/helper_gl.h
    src/utility/helper_image.h
    src/utility/helper_logger.h
    src/utility/helper_math.h
    src/utility/helper_string.h
    src/utility/helper_timer.h	
	src/utility/memory_arena.h
    src/utility/timer.h    
	src/utility/types.h

	src/renderer/bsdfs/bxdf.h
	src/renderer/bsdfs/optical.h
	src/renderer/bsdfs/lambert.h
	src/renderer/bsdfs/specular.h
	src/renderer/bsdfs/microfacet.h
	src/renderer/loaders/loader.h
	src/renderer/loaders/mitsubaloader.h
	src/renderer/loaders/mitsubaloader.cpp
	src/renderer/loaders/objloader.h
	src/renderer/loaders/objloader.cpp
	src/renderer/loaders/propertylist.h
	src/renderer/loaders/propertylist.cpp
	src/renderer/aabb.h
	src/renderer/bsdf.h
	src/renderer/bvh.h
	src/renderer/camera.h
	src/renderer/environment.cpp
	src/renderer/environment.h
	src/renderer/framebuffer.cpp
	src/renderer/framebuffer.h
	src/renderer/fwd.h	
	src/renderer/imageio.h
	src/renderer/imageio.cpp
	src/renderer/material.h
	src/renderer/pathtracer.cpp
	src/renderer/pathtracer.h
	src/renderer/pathtracer_kernel.h
	src/renderer/ray.h
	src/renderer/rayqueue.h
	src/renderer/sampling.h
	src/renderer/scene.cpp
	src/renderer/scene.h
	src/renderer/spectrum.h
	src/renderer/texture.cpp
	src/renderer/texture.h
	src/renderer/transform.cpp
	src/renderer/transform.h
	src/renderer/triangle.h
	src/renderer/triangle.cpp
	src/renderer/vertex.h

    src/main.cpp    
    )

########################################
# Sort the file into different folders
foreach(_source_file IN ITEMS ${sources})
    get_filename_component(_source_path "${_source_file}" PATH)
    string(REPLACE "${CMAKE_SOURCE_DIR}" "" _group_path "${_source_path}")
    string(REPLACE "/" "\\" _group_path "${_group_path}")
    source_group("${_group_path}" FILES "${_source_file}")
endforeach()

########################################
# Add Executable

 
cuda_add_executable(renderer ${sources})
set_target_properties(renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(renderer
        ${LIBRARIES})