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
find_package(OpenGL REQUIRED)

if(UNIX)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY})
else(UNIX)
    set(EXTERNAL "external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})   
endif(UNIX)

set(GLM_ROOT_DIR "external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

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
    src/ext/tinyobjloader/tiny_obj_loader.h
    src/ext/tinyobjloader/tiny_obj_loader.cc    
    src/GUI/gui.cpp
    src/GUI/gui.h    
    src/utility/exception.h
    src/utility/helper_cuda.h
    src/utility/helper_functions.h
    src/utility/helper_gl.h
    src/utility/helper_image.h
    src/utility/helper_logger.h
    src/utility/helper_math.h
    src/utility/helper_string.h
    src/utility/helper_timer.h    
    src/utility/timer.h    
    src/renderer/core/api.cpp
    src/renderer/core/api.h
	src/renderer/core/bsdf.h
    src/renderer/core/camera.cpp
    src/renderer/core/camera.h
	src/renderer/core/cpurender.h
    src/renderer/core/film.cpp
    src/renderer/core/film.h
    src/renderer/core/fwd.h
    src/renderer/core/geometry.h
	src/renderer/core/integrator.cpp
    src/renderer/core/integrator.h
    src/renderer/core/interaction.h
    src/renderer/core/light.cpp
    src/renderer/core/light.h
    src/renderer/core/material.cpp
    src/renderer/core/material.h
    src/renderer/core/medium.cpp
    src/renderer/core/medium.h
    src/renderer/core/parameterset.cpp
    src/renderer/core/parameterset.h
    src/renderer/core/primitive.h
    src/renderer/core/renderer.h
    src/renderer/core/sampler.h
    src/renderer/core/sampling.h
    src/renderer/core/scene.cpp
    src/renderer/core/scene.h
    src/renderer/core/spectrum.cpp
    src/renderer/core/spectrum.h
    src/renderer/core/transform.cpp
    src/renderer/core/transform.h
    src/renderer/core/triangle.cpp
    src/renderer/core/triangle.h
	src/renderer/kernel/cudarenderer.h
	src/renderer/kernel/cudascene.cpp
	src/renderer/kernel/cudascene.h
    src/renderer/kernel/kernel.cu
    src/renderer/loader/sceneloader.h
    src/renderer/loader/sceneloader.cpp
    src/renderer/loader/mitsubaloader.h
    src/renderer/loader/mitsubaloader.cpp
    src/renderer/loader/pbrtloader.h
    src/renderer/loader/pbrtloader.cpp
    src/renderer/loader/objloader.h     
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