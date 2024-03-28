set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cmake_minimum_required(VERSION 3.24)
project(bench_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# message(STATUS "Torch include: ${TORCH_INCLUDE_DIRS}")
# message(STATUS "Torch CXX flags: ${TORCH_CXX_FLAGS}")

# find_package(CuDNN QUIET REQUIRED)

set(PROJECT_CUDA_ARCH native CACHE STRING "CUDA ARCH")
set(CMAKE_CUDA_ARCHITECTURES ${PROJECT_CUDA_ARCH})

list(APPEND CUTLASS_CUDA_NVCC_FLAGS "--expt-relaxed-constexpr" "--std=c++17")
list(APPEND CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "${CUTLASS_CUDA_NVCC_FLAGS}" "${ARCH_FLAGS}" "-O3" "--use_fast_math" "-lineinfo" "-Xptxas=-v" "--disable-warnings")

include_directories("../third_party/cutlass/include")
include_directories("../csrc/")


message(STATUS "final CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "final CUDA NVCC flags: ${CUDA_NVCC_FLAGS}")

add_executable(attention attention.cu)
target_link_libraries(attention 
    ${CUDA_LIBRARIES}
    # ${TORCH_LIBRARIES}
    # ${CUDA_CUBLAS_LIBRARIES}
)

target_compile_options(attention PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
