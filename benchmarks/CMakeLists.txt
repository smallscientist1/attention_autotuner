cmake_minimum_required(VERSION 3.18)
project(bench_attention)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

find_package(CUDA QUIET REQUIRED)
# find_package(CuDNN QUIET REQUIRED)

cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ARCH_FLAGS}")
message(STATUS "CUDA Architecture flags = ${ARCH_FLAGS}")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

list(APPEND CUTLASS_CUDA_NVCC_FLAGS "--expt-relaxed-constexpr" "--std=c++17")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUTLASS_CUDA_NVCC_FLAGS} ${ARCH_FLAGS} -O3 --use_fast_math -lineinfo -Xptxas=-v --disable-warnings")

include_directories("../third_party/cutlass/include")
include_directories("../csrc/")


cuda_add_executable(attention attention.cu)
target_link_libraries(attention 
    ${CUDA_LIBRARIES}
    # ${CUDA_CUBLAS_LIBRARIES}
)
