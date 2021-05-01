add_library(CUDA_11
        src/helper_multiprocess.cpp
        src/multithreading.cpp
        )
target_include_directories(CUDA_11 PUBLIC inc)
set_target_properties(CUDA_11 PROPERTIES LINKER_LANGUAGE CXX)