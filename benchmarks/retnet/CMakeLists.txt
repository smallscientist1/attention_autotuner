set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cmake_minimum_required(VERSION 3.24) # support arch native # 3.18
project(bench_retnet LANGUAGES CXX CUDA)

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

# must delete CMakeCache.txt(命令行指定的参数会缓存, CMAkeLists中的不会;通过 set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA ARCH") 初始化缓存变量)
set(PROJECT_CUDA_ARCH native CACHE STRING "CUDA ARCH")
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES) # 默认定义为52
set(CMAKE_CUDA_ARCHITECTURES ${PROJECT_CUDA_ARCH})

list(APPEND CUTLASS_CUDA_NVCC_FLAGS "--expt-relaxed-constexpr" "--std=c++17")
list(APPEND CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "${CUTLASS_CUDA_NVCC_FLAGS}" "${ARCH_FLAGS}" "-O3" "--use_fast_math" "-lineinfo" "-Xptxas=-v" "--disable-warnings")

include_directories("../../third_party/cutlass/include")
include_directories("../../csrc/")


message(STATUS "final CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "final CUDA NVCC flags: ${CUDA_NVCC_FLAGS}")

add_executable(retnet retnet.cu) # 此处Cmake将CMAKE_CUDA_ARCHITECTURES 赋值给TARGET_PROPERTY

target_link_libraries(retnet
    ${CUDA_LIBRARIES}
    # ${TORCH_LIBRARIES}
    # ${CUDA_CUBLAS_LIBRARIES}
)

# conditional expression
target_compile_options(retnet PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
# set_property(TARGET retnet PROPERTY CUDA_ARCHITECTURES 80)
# set_property(TARGET retnet PROPERTY CUDA_ARCHITECTURES native)