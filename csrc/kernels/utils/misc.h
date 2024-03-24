#pragma once
#include "cutlass/numeric_conversion.h"

#include "cute/layout.hpp"
#include "cute/tensor.hpp"

#define CUDA_SAFE_CALL(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        cudaError_t result = (x);                                                                  \
        if (result != cudaSuccess)                                                                 \
        {                                                                                          \
            const char* msg = cudaGetErrorString(result);                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #x " failed with error"                                    \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
    
namespace cute{

template<typename Layout>
inline __device__ auto convert_layout_rowcol_Aregs(Layout rowcol_layout){
  using namespace cute;
  static_assert(decltype(size<0, 0>(rowcol_layout))::value == 2);
  static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);
  auto l = logical_divide(rowcol_layout, Shape<Underscore, Shape<Underscore, Int<2>>>{});  // ((2, MMA_M), (2, (2, MMA_N / 2)))
  return make_layout(make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)), get<0>(get<1>(get<1>(l)))),
                       get<1>(get<0>(l)),
                       get<1>(get<1>(get<1>(l))));
}

  inline __device__ auto convert_layout_C_Aregs(){
  using namespace cute;
  auto layout_s = Layout<Shape<Shape<_2,_2>,_2,_16>>{};
  auto l = logical_divide(layout_s,Shape<Underscore, Underscore,_2>{});
  /*if(threadIdx.x==0 && blockIdx.x==0){
    print(l.layout(),"\nl_layout\n");
    print(make_layout(make_layout(get<0>(get<0>(l)),get<1>(get<0>(l)),get<0>(get<2>(l))),
                     get<1>(l),
                     get<1>(get<2>(l))));
  }*/
  return make_layout(make_layout(get<0>(get<0>(l)),get<1>(get<0>(l)),get<0>(get<2>(l))),
                     get<1>(l),
                     get<1>(get<2>(l)));

  // return Layout<Shape<Shape<_2,_2,_2>,_2,_8>>{};
}

template<class LayoutType>
inline __device__ auto convert_layout_scores(LayoutType layout_s){
  using namespace cute;
  static_assert(decltype(size<0>(layout_s))::value == 4);
  static_assert(decltype(rank(layout_s))::value == 3);
  // auto layout_s = Layout<Shape<Shape<_2,_2>,_2,_16>>{};
  auto l = logical_divide(layout_s, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
  return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)), make_layout(get<0>(get<0>(l)), get<2>(l)));
}

template<int ATOMNUM,class LayoutType>
inline __device__ auto convert_layout_scores_copyview(LayoutType layout_s){
  using namespace cute;
  // (2,8)
  auto l = logical_divide(layout_s, Shape<Underscore,Int<ATOMNUM>>{});
  return make_layout(get<0>(get<1>(l)),get<0>(l),get<1>(get<1>(l)));
}


}

extern "C" void cuda_init()
{
// CUDA_SAFE_CALL(cudaDeviceReset());
// total memory:570687488
CUDA_SAFE_CALL(cudaSetDevice(0));
// create streams/handles
}
extern "C" void cuda_free()
{
CUDA_SAFE_CALL(cudaSetDevice(0));
}

inline __device__ longlong4 make_int8(int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7) {
  int2 i0 = make_int2(x0, x1);
  int2 i1 = make_int2(x2, x3);
  int2 i2 = make_int2(x4, x5);
  int2 i3 = make_int2(x6, x7);
  long long l0 = *(long long*)&i0;
  long long l1 = *(long long*)&i1;
  long long l2 = *(long long*)&i2;
  long long l3 = *(long long*)&i3;
  return make_longlong4(l0, l1, l2, l3);
}

inline __device__ longlong4 make_longlong4(int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7) {
  return make_int8(x0, x1, x2, x3, x4, x5, x6, x7);
}

inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}
#ifndef __HALF_COMPARE_EX__
#define __HALF_COMPARE_EX__
inline __device__ half max(half x, half y) { return x > y ? x : y; }
inline __device__ half min(half x, half y) { return x < y ? x : y; }
#endif

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
inline __device__ half HALF_MATH_NAME(half x, half y) {                   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
inline __device__ half HALF_MATH_NAME(half x) {                          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY
