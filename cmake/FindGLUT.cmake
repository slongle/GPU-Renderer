#
# Find GLUT
#
# Try to find GLUT library.
# This module defines the following variables:
# - GLUT_INCLUDE_DIRS
# - GLUT_LIBRARIES
# - GLUT_FOUND
#
# The following variables can be set as arguments for the module.
# - GLUT_ROOT_DIR : Root library directory of GLUT
# - GLUT_USE_STATIC_LIBS : Specifies to use static version of GLFW library (Windows only)
#
# References:
# - https://github.com/progschj/OpenGL-Examples/blob/master/cmake_modules/FindGLFW.cmake
# - https://bitbucket.org/Ident8/cegui-mk2-opengl3-renderer/src/befd47200265/cmake/FindGLFW.cmake
#

# Additional modules
include(FindPackageHandleStandardArgs)

if (WIN32)
    # Find include files
	find_path(
		GLUT_INCLUDE_DIR 
        NAMES GL/freeglut.h
        PATHS
		$ENV{PROGRAMFILES}/include
		${GLUT_ROOT_PATH}/include
		${PROJECT_SOURCE_DIR}/external/include
		DOC "The directory where GL/freeglut.h resides")



	# Use glut.lib for static library
	if (GLUT_USE_STATIC_LIBS)
		set(GLUT_LIBRARY_NAME freeglut)
	else()
		set(GLUT_LIBRARY_NAME freeglutdll)
	endif()

    MESSAGE(STATUS ${GLUT_LIBRARY_NAME})
    MESSAGE(STATUS ${GLUT_ROOT_DIR}/lib)

	# Find library files
	find_library(
		GLUT_LIBRARY
		NAMES ${GLUTL_LIBRARY_NAME}
		PATHS
		$ENV{PROGRAMFILES}/lib
		${GLUT_ROOT_DIR}/lib
        ${PROJECT_SOURCE_DIR}/external/lib)

    
	unset(GLUT_LIBRARY_NAME)
else()
endif()

# Handle REQUIRD argument, define *_FOUND variable
find_package_handle_standard_args(GLUT DEFAULT_MSG GLUT_INCLUDE_DIR GLUT_LIBRARY)

# Define GLUT_LIBRARIES and GLUT_INCLUDE_DIRS
if (GLUT_FOUND)
	set(GLUT_LIBRARIES ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
	set(GLUT_INCLUDE_DIRS ${GLUT_INCLUDE_DIR})
endif()

# Hide some variables
mark_as_advanced(GLUT_INCLUDE_DIR GLUT_LIBRARY)
