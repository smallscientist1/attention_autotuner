#pragma once
#include "cutlass/numeric_conversion.h"

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/algorithm/copy.hpp"

namespace cute {

template <int N>
CUTE_HOST_DEVICE
void cp_async_wait_flash() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

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


}

