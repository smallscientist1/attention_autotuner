#include "cutlass/numeric_conversion.h"

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/copy.hpp"

namespace cute{

// only for D=512, L2cache
template<class GmemTiledCopy,class GTensor1, class RTensor1>
__device__ void copy_Global2Reg(GmemTiledCopy gmem_tiled_copy_QKV, 
              GTensor1& gQ_partition, RTensor1& rQ_partition){
                CUTE_STATIC_ASSERT_V(size<0>(gQ_partition)==size<0>(rQ_partition));
                CUTE_STATIC_ASSERT_V(size<1>(gQ_partition)==size<2>(rQ_partition));
                #pragma unroll
                for(int i=0;i<size<1>(gQ_partition);i++){
                  cute::copy(gmem_tiled_copy_QKV, gQ_partition(_, i, _0{}), rQ_partition(_, _0{},i));
                }
}

template<class GmemTiledCopy,class GTensor1, class RTensor1>
__device__ void copy_Reg2Global(GmemTiledCopy gmem_tiled_copy_QKV, 
              RTensor1& rQ_partition, GTensor1& gQ_partition){
                CUTE_STATIC_ASSERT_V(size<0>(gQ_partition)==size<0>(rQ_partition));
                CUTE_STATIC_ASSERT_V(size<1>(gQ_partition)==size<2>(rQ_partition));
                #pragma unroll
                for(int i=0;i<size<1>(gQ_partition);i++){
                  cute::copy(gmem_tiled_copy_QKV, rQ_partition(_, _0{},i), gQ_partition(_, i, _0{}));
                }
}

template<class GmemTiledCopy,class GTensor1, class STensor1,class GTensor2, class STensor2>
class CopyAsyncQK_g2s{
public:
  __device__ CopyAsyncQK_g2s(GmemTiledCopy gmem_tiled_copy_QKV, 
              GTensor1& gQ_partition, STensor1& sQ_partition, 
              GTensor2& gK_partition, STensor2& sK_partition,
              int gQ_stride, int sQ_stride, int gK_stride, int sK_stride, int num_stage=2): 
              gmem_tiled_copy_QKV(gmem_tiled_copy_QKV),
              gQ_partition(gQ_partition), sQ_partition(sQ_partition),
              gK_partition(gK_partition), sK_partition(sK_partition),
              gQ_stride(gQ_stride), sQ_stride(sQ_stride), gK_stride(gK_stride), sK_stride(sK_stride),
              cur_iter(0), num_stage(num_stage)
              {
              }
  inline __device__ void prologue(){
    // cur_iter = 0;
    #pragma unroll
    for (int m = 0; m < size<1>(gQ_partition); ++m) {
          #pragma unroll
          for (int k = 0; k < size<2>(gQ_partition); ++k) {
            cute::copy(gmem_tiled_copy_QKV, gQ_partition(_, m, k), sQ_partition(_, m, k));
          }
    }
    #pragma unroll
    for (int m = 0; m < size<1>(gK_partition); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(gK_partition); ++k) {
          cute::copy(gmem_tiled_copy_QKV, gK_partition(_, m, k), sK_partition(_, m, k));
        }
    }
    cute::cp_async_fence();
    gQ_partition.data() = gQ_partition.data() + gQ_stride;
    gK_partition.data() = gK_partition.data() + gK_stride;
    sQ_partition.data() = sQ_partition.data() + sQ_stride;
    sK_partition.data() = sK_partition.data() + sK_stride;
    if((cur_iter+1)%num_stage==0){
      sK_partition.data() = sK_partition.data() + (-sK_stride*num_stage);
      sQ_partition.data() = sQ_partition.data() + (-sQ_stride*num_stage);
    }
    cur_iter++;
  }
  inline __device__ void body(){
    #pragma unroll
    for (int m = 0; m < size<1>(gQ_partition); ++m) {
          #pragma unroll
          for (int k = 0; k < size<2>(gQ_partition); ++k) {
            cute::copy(gmem_tiled_copy_QKV, gQ_partition(_, m, k), sQ_partition(_, m, k));
          }
    }
    #pragma unroll
    for (int m = 0; m < size<1>(gK_partition); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(gK_partition); ++k) {
          cute::copy(gmem_tiled_copy_QKV, gK_partition(_, m, k), sK_partition(_, m, k));
        }
    }
    cute::cp_async_fence();
    gQ_partition.data() = gQ_partition.data() + gQ_stride;
    gK_partition.data() = gK_partition.data() + gK_stride;
    sK_partition.data() = sK_partition.data() + (sK_stride);
    sQ_partition.data() = sQ_partition.data() + (sQ_stride);
    if((cur_iter+1)%num_stage==0){
      sK_partition.data() = sK_partition.data() + (-sK_stride*num_stage);
      sQ_partition.data() = sQ_partition.data() + (-sQ_stride*num_stage);
    }
    cur_iter++;
  }
  inline __device__ void epilogue(){
    // TODO
    #pragma unroll
    for (int m = 0; m < size<1>(gQ_partition); ++m) {
          #pragma unroll
          for (int k = 0; k < size<2>(gQ_partition); ++k) {
            cute::copy(gmem_tiled_copy_QKV, gQ_partition(_, m, k), sQ_partition(_, m, k));
          }
    }
    #pragma unroll
    for (int m = 0; m < size<1>(gK_partition); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(gK_partition); ++k) {
          cute::copy(gmem_tiled_copy_QKV, gK_partition(_, m, k), sK_partition(_, m, k));
        }
    }
    cute::cp_async_fence();
  }

private:
  int cur_iter;
  GmemTiledCopy gmem_tiled_copy_QKV;
  GTensor1& gQ_partition;
  STensor1& sQ_partition;
  GTensor2& gK_partition;
  STensor2& sK_partition;
  int gQ_stride, sQ_stride, gK_stride, sK_stride;
  int num_stage;
};

template<class GmemTiledCopy,class GTensor1, class STensor1>
class CopyAsyncV_g2s{
public:
  __device__ CopyAsyncV_g2s(GmemTiledCopy gmem_tiled_copy_QKV, 
              GTensor1& gV_partition, STensor1& sV_partition, 
              int gV_stride, int sV_stride, int num_stage=2): 
              gmem_tiled_copy_QKV(gmem_tiled_copy_QKV),
              gV_partition(gV_partition), sV_partition(sV_partition),
              gV_stride(gV_stride), sV_stride(sV_stride),
              cur_iter(0), num_stage(num_stage)
              {
              }
  inline __device__ void prologue(){
    // cur_iter = 0;
    #pragma unroll
    for (int m = 0; m < size<1>(gV_partition); ++m) {
          #pragma unroll
          for (int k = 0; k < size<2>(gV_partition); ++k) {
            cute::copy(gmem_tiled_copy_QKV, gV_partition(_, m, k), sV_partition(_, m, k));
          }
    }
    cute::cp_async_fence();
    gV_partition.data() = gV_partition.data() + gV_stride;
    sV_partition.data() = sV_partition.data() + sV_stride;
    if((cur_iter+1)%num_stage==0){
      sV_partition.data() = sV_partition.data() + (-sV_stride*num_stage);
    }
    cur_iter++;
  }
  inline __device__ void body(){
    #pragma unroll
    for (int m = 0; m < size<1>(gV_partition); ++m) {
          #pragma unroll
          for (int k = 0; k < size<2>(gV_partition); ++k) {
            cute::copy(gmem_tiled_copy_QKV, gV_partition(_, m, k), sV_partition(_, m, k));
          }
    }
    cute::cp_async_fence();
    gV_partition.data() = gV_partition.data() + gV_stride;
    sV_partition.data() = sV_partition.data() + sV_stride;

    if((cur_iter+1)%num_stage==0){
      sV_partition.data() = sV_partition.data() + (-sV_stride*num_stage);
    }
    cur_iter++;
  }
  inline __device__ void epilogue(){
    // TODO
    #pragma unroll
    for (int m = 0; m < size<1>(gV_partition); ++m) {
          #pragma unroll
          for (int k = 0; k < size<2>(gV_partition); ++k) {
            cute::copy(gmem_tiled_copy_QKV, gV_partition(_, m, k), sV_partition(_, m, k));
          }
    }
    cute::cp_async_fence();
  }
private:
  int cur_iter;
  GmemTiledCopy gmem_tiled_copy_QKV;
  GTensor1& gV_partition;
  STensor1& sV_partition;
  int gV_stride, sV_stride;
  int num_stage;
};

template<class SmemTiledCopy1, class STensor1, class RTensor1,
          class SmemTiledCopy2, class STensor2, class RTensor2,
          class TiledMMAType, class RTensor3, class RTensor4, class RTensor5>
class MatmulQK_s2r{
public:
  __device__ MatmulQK_s2r(
    SmemTiledCopy1 smem_tiled_copy_Q, STensor1& sQ_copypartition, RTensor1& rQ_copy_view, 
    SmemTiledCopy2 smem_tiled_copy_K, STensor2& sK_copypartition, RTensor2& rK_copy_view, 
    TiledMMAType tiled_mma1, RTensor3& rQ, RTensor4& rK, RTensor5& acc_s_fragment, 
    int sQ_stride, int sK_stride, int num_stage=2      
  ):
  smem_tiled_copy_Q(smem_tiled_copy_Q), sQ_copypartition(sQ_copypartition), rQ_copy_view(rQ_copy_view), 
  smem_tiled_copy_K(smem_tiled_copy_K), sK_copypartition(sK_copypartition), rK_copy_view(rK_copy_view), 
  tiled_mma1(tiled_mma1), rQ(rQ), rK(rK), acc_s_fragment(acc_s_fragment), 
  sQ_stride(sQ_stride), sK_stride(sK_stride), 
  cur_iter(0), num_stage(num_stage)
  {
  }
  inline __device__ void prologue(){
    // TODO
    cur_iter = 0;
    cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, _0{}), rQ_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, _0{}), rK_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(rK); ++i) {
      if(i < size<2>(rK) - 1){
        cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, i+1), rQ_copy_view(_, _, i+1));
        cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, i+1), rK_copy_view(_, _, i+1));
      }
      cute::gemm(tiled_mma1, rQ(_, _, i), rK(_, _, i), acc_s_fragment);
    }
    sQ_copypartition.data() = sQ_copypartition.data() + sQ_stride;
    sK_copypartition.data() = sK_copypartition.data() + sK_stride;
    cur_iter++;
  }
  inline __device__ void body(){
    cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, _0{}), rQ_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, _0{}), rK_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(rK); ++i) {
      if(i < size<2>(rK) - 1){
        cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, i+1), rQ_copy_view(_, _, i+1));
        cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, i+1), rK_copy_view(_, _, i+1));
      }
      cute::gemm(tiled_mma1, rQ(_, _, i), rK(_, _, i), acc_s_fragment);
    }
    sQ_copypartition.data() = sQ_copypartition.data() + sQ_stride;
    sK_copypartition.data() = sK_copypartition.data() + sK_stride;
    if((cur_iter+1)%num_stage==0){
      sQ_copypartition.data() = sQ_copypartition.data() + (-sQ_stride*num_stage);
      sK_copypartition.data() = sK_copypartition.data() + (-sK_stride*num_stage);
    }
    cur_iter++;
  }
  inline __device__ void epilogue(){
    cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, _0{}), rQ_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, _0{}), rK_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(rK); ++i) {
      if(i < size<2>(rK) - 1){
        cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, i+1), rQ_copy_view(_, _, i+1));
        cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, i+1), rK_copy_view(_, _, i+1));
      }
      cute::gemm(tiled_mma1, rQ(_, _, i), rK(_, _, i), acc_s_fragment);
    }
    sQ_copypartition.data() = sQ_copypartition.data() + sQ_stride;
    sK_copypartition.data() = sK_copypartition.data() + sK_stride;
    if((cur_iter+1)%num_stage==0){
      sQ_copypartition.data() = sQ_copypartition.data() + (-sQ_stride*num_stage);
      sK_copypartition.data() = sK_copypartition.data() + (-sK_stride*num_stage);
    }
    cur_iter++; // cur_iter=0;
  }

private:
  int cur_iter;
  SmemTiledCopy1 smem_tiled_copy_Q;
  STensor1& sQ_copypartition;
  RTensor1& rQ_copy_view;
  SmemTiledCopy2 smem_tiled_copy_K;
  STensor2& sK_copypartition;
  RTensor2& rK_copy_view;
  TiledMMAType tiled_mma1;
  RTensor3& rQ;
  RTensor4& rK;
  RTensor5& acc_s_fragment;
  int sQ_stride, sK_stride;
  int num_stage;
};

template<class SmemTiledCopy2, class STensor2, class RTensor2,
          class TiledMMAType, class RTensor4, class RTensor5>
class MatmulV_s2r{
public:
  __device__ MatmulV_s2r(SmemTiledCopy2 smem_tiled_copy_V, STensor2& sV_copypartition, RTensor2& rV_copy_view,
                    TiledMMAType tiled_mma, /*RTensor3& rP,*/ RTensor4& rV, RTensor5& acc_o_fragment,
                    int sV_stride, int num_stage=2):
    smem_tiled_copy_V(smem_tiled_copy_V), sV_copypartition(sV_copypartition), rV_copy_view(rV_copy_view),
    tiled_mma(tiled_mma), /*rP(rP),*/ rV(rV), acc_o_fragment(acc_o_fragment),
    sV_stride(sV_stride),
    cur_iter(0), cur_iter_sv(0), num_stage(num_stage)
    {
    }
  template<class RTensor3>
  inline __device__ void prologue(RTensor3& rP){
    // TODO
    cur_iter = 0;
    cute::copy(smem_tiled_copy_V, sV_copypartition(_, _, _0{}), rV_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(rV); ++i) {
      if(i < size<2>(rV) - 1){
        cute::copy(smem_tiled_copy_V, sV_copypartition(_, _, i+1), rV_copy_view(_, _, i+1));
      }
      cute::gemm(tiled_mma, rP(_, _, cur_iter*size<2>(rV) +i), rV(_, _, i), acc_o_fragment);
    }
    sV_copypartition.data() = sV_copypartition.data() + sV_stride;
    cur_iter++;
  }
  template<class RTensor3>
  inline __device__ void body(RTensor3& rP){
    cute::copy(smem_tiled_copy_V, sV_copypartition(_, _, _0{}), rV_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(rV); ++i) {
      if(i < size<2>(rV) - 1){
        cute::copy(smem_tiled_copy_V, sV_copypartition(_, _, i+1), rV_copy_view(_, _, i+1));
      }
      cute::gemm(tiled_mma, rP(_, _, cur_iter_sv*size<2>(rV) +i), rV(_, _, i), acc_o_fragment);
    }
    sV_copypartition.data() = sV_copypartition.data() + sV_stride;
    if((cur_iter+1)%num_stage==0){
      sV_copypartition.data() = sV_copypartition.data() + (-sV_stride*num_stage);
    }
    cur_iter++;
    cur_iter_sv++;
  }
  template<class RTensor3>
  inline __device__ void epilogue(RTensor3& rP){
    cute::copy(smem_tiled_copy_V, sV_copypartition(_, _, _0{}), rV_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(rV); ++i) {
      if(i < size<2>(rV) - 1){
        cute::copy(smem_tiled_copy_V, sV_copypartition(_, _, i+1), rV_copy_view(_, _, i+1));
      }
      cute::gemm(tiled_mma, rP(_, _, cur_iter_sv*size<2>(rV) +i), rV(_, _, i), acc_o_fragment);
    }
    sV_copypartition.data() = sV_copypartition.data() + sV_stride;
    if((cur_iter+1)%num_stage==0){
      sV_copypartition.data() = sV_copypartition.data() + (-sV_stride*num_stage);
    }
    cur_iter++; // cur_iter=0;
    cur_iter_sv=0;
  }
private:
  int cur_iter, cur_iter_sv;
  SmemTiledCopy2 smem_tiled_copy_V;
  STensor2& sV_copypartition;
  RTensor2& rV_copy_view;
  TiledMMAType tiled_mma;
  // RTensor3& rP;
  RTensor4& rV;
  RTensor5& acc_o_fragment;
  int sV_stride;
  int num_stage;
};

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