cmake_minimum_required(VERSION 3.16)
project(ParallelApplications2 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE OFF)

add_subdirectory(../libs/ [binary_dir])

add_library(FreeImage STATIC IMPORTED)
set_property(TARGET FreeImage PROPERTY IMPORTED_LOCATION "/usr/lib/libfreeimage.a")

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include_directories( ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})

add_executable(PA2
        main.cu
        HeightMap.cuh   HeightMap.cu
        Particles.cuh   Particles.cu
        Settings.h      Settings.cpp
        )

set_target_properties(PA2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(PA2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode arch=compute_35,code=sm_35;
        -Wno-deprecated-gpu-targets;
        >)

target_link_libraries(PA2
        PG
        CUDA_11
        FreeImage
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARIES}
        ${GLEW_LIBRARIES}
        cublas
        )
set_property(TARGET PA2 PROPERTY CUDA_ARCHITECTURES 35 50 72)






