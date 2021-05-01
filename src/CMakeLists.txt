cmake_minimum_required(VERSION 3.19)
project(PA2Project LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE OFF)

add_library(FreeImage STATIC IMPORTED)
set_property(TARGET FreeImage PROPERTY IMPORTED_LOCATION "/usr/lib/libfreeimage.a")

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

add_subdirectory(../libs [binary_dir])

include_directories( ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})

add_executable(PA2Project main.cu
        Settings.cpp Settings.h HeightMap.cu HeightMap.cuh Particles.cu Particles.cuh)

target_link_libraries(PA2Project
        PG
        CUDA_11
        FreeImage
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARIES}
        ${GLEW_LIBRARIES}
        cublas
        )

set_target_properties(
        PA2Project
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)