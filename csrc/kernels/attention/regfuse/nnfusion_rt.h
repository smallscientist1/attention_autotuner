#pragma once
#include <cuda_fp16.h>
#include <mma.h>


// extern "C" int kernel_entry(half* Parameter_0_0_0, half* Parameter_1_0_0, half* Parameter_2_0_0, half* Result_7_0_0, int B=4, int H=8, int Seq_k=2048,int Seq_q=2048);

#include "welder_cuda.h"
#include <stdexcept>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#ifndef __HALF_COMPARE_EX__
#define __HALF_COMPARE_EX__
inline __device__ half hmax(half x, half y) { return x > y ? x : y; }
inline __device__ half hmin(half x, half y) { return x < y ? x : y; }
#endif
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
#include <stdint.h>

constexpr float softmax_scale = 1.250000e-01f;

/*
// 128*384
constexpr int Br = 128;
constexpr int Bc = 64; // 32
constexpr int Kd = 256;
constexpr int D = 256;

constexpr bool unrollLastIter = true;
// for sQ,sK,sV,sO swizzle
constexpr int SmemKAtom = 64;
constexpr int kSwizzle = SmemKAtom == 32 ? 2 : 3;
// for q&k splitk
__device__ constexpr int BlockKSmem = 256; // avoid 'error: identifier "BlockKSmem" is undefined in device code'
constexpr int num_stages_qk = 1;
constexpr bool load_q_once = (BlockKSmem == Kd);
// for V splitk
constexpr int BlockKSmem2 = 64; // 32
constexpr int num_stages_v = 1;
constexpr int shared_matmulqkv = num_stages_qk*(Br)*BlockKSmem*sizeof(half)+num_stages_qk*Bc*BlockKSmem*sizeof(half)+num_stages_v*BlockKSmem2*D*sizeof(half);
constexpr int shared_out = Br * D * sizeof(half);
constexpr int shared_mem = (shared_matmulqkv) > shared_out ? (shared_matmulqkv):shared_out;//(acc_o(p(q,k),v))

constexpr int Nthreads = 256;

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
*/


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
// m_old, m_new, acc_o, p, v


template<class LayoutType>
inline __device__ auto convert_layout_scores(LayoutType layout_s){
  using namespace cute;
  static_assert(decltype(size<0>(layout_s))::value == 4);
  static_assert(decltype(rank(layout_s))::value == 3);
  // auto layout_s = Layout<Shape<Shape<_2,_2>,_2,_16>>{};
  auto l = logical_divide(layout_s, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
  return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)), make_layout(get<0>(get<0>(l)), get<2>(l)));
}

template <int N>
CUTE_HOST_DEVICE
void cp_async_wait_flash() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

struct MaxOp_float {
// This is slightly faster
__device__ inline float operator()(float const &x, float const &y) { return max(x, y); } // cute::max(x,y)
};
template<typename T>
struct SumOp {
__device__ inline T operator()(T const & x, T const & y) { return x + y; }
};
template<typename T>
struct SumAbsOp {
__device__ inline T operator()(T const & x, T const & y) { return x + abs(y); }
};

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ inline T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};
template<>
struct Allreduce<2> {
template<typename T, typename Operator> 
static __device__ inline T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void thread_reduce_(cute::Tensor<Engine0, Layout0> const &tensor, cute::Tensor<Engine1, Layout1> &summary, Operator &op) {
  using namespace cute;
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? op(0, tensor(mi, 0)) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}
template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void quad_allreduce_(cute::Tensor<Engine0, Layout0> &dst, cute::Tensor<Engine1, Layout1> &src, Operator &op) {
    using namespace cute;
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void reduce_(cute::Tensor<Engine0, Layout0> const& tensor, cute::Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_max(cute::Tensor<Engine0, Layout0> const& tensor, cute::Tensor<Engine1, Layout1> &max){
    MaxOp_float max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_sum(cute::Tensor<Engine0, Layout0> const& tensor, cute::Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    reduce_(tensor, sum, sum_op);
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_sumabs(cute::Tensor<Engine0, Layout0> const& tensor, cute::Tensor<Engine1, Layout1> &sum){
    SumAbsOp<float> sumabs_op;
    reduce_(tensor, sum, sumabs_op);
}

template<class Tensor0, class Tensor1, class Tensor2>
__device__ inline void update_r(Tensor0& r_new_fragment, Tensor1& r_wo_clamp_fragment, Tensor2& scores){
  using namespace cute;
    // r_wo_clamp
    Tensor r_wo_clamp_fragment_tmp = make_fragment_like(r_wo_clamp_fragment);
    reduce_sumabs(scores, r_wo_clamp_fragment_tmp);
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(r_wo_clamp_fragment);ax0++){
      r_wo_clamp_fragment(ax0) += r_wo_clamp_fragment_tmp(ax0);
    }
    // r_new = max(r_wo_clamp, 1)
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(r_wo_clamp_fragment);ax0++){
      r_new_fragment(ax0) = max(r_wo_clamp_fragment(ax0), 1.0f);
    }
} 

template<class SmemTiledCopy1, class STensor1, class RTensor1,class Tensor0, class Tensor1>
__device__ inline void multiply_mask(SmemTiledCopy1 smem_tiled_copy_mask, STensor1& sMask_copypartition, RTensor1& rMask_copy_view, 
                            Tensor0& acc_s_fragment, Tensor1& rMask){
  using namespace cute;
  // qk*m
    cute::copy(smem_tiled_copy_mask, sMask_copypartition(_,_,_0{}), rMask_copy_view(_,_,_0{}));
    #pragma unroll
    for(int ax0 = 0;ax0 < size<2>(acc_s_fragment);ax0++){
      if(ax0 < size<2>(acc_s_fragment)-1){
        cute::copy(smem_tiled_copy_mask, sMask_copypartition(_,_,ax0+1), rMask_copy_view(_,_,ax0+1));
      }
      #pragma unroll
      for(int ax1 = 0;ax1 < size<1>(acc_s_fragment);ax1++){
        #pragma unroll
        for(int ax2 = 0;ax2 < size<0>(acc_s_fragment);ax2++){
          acc_s_fragment(ax2,ax1,ax0) = acc_s_fragment(ax2,ax1,ax0) * __half2float(rMask(ax2,ax1,ax0));
        }
      }
    }
}

template<int Kd,int D, int Br,int Bc,int Nthreads,int BlockKSmem=Kd, int num_stages_qk=1, bool load_q_once=true, int BlockKSmem2=Bc, int num_stages_v=1,int SmemKAtom=64,int kSwizzle=3,bool unrollLastIter=true>
__global__ void __launch_bounds__(Nthreads) flashattn_fwd_regfuse(half* Parameter_0_0_0, half* Parameter_1_0_0, half* Parameter_2_0_0, half* Result_7_0_0, int H, int seq_k, int seq_q){

    extern __shared__ char shared[];

    // constexpr int seq_k = 2048;
    // constexpr int seq_q = 2048;
    int Tc = seq_k/Bc; 
    int Tr = seq_q/Br; 
    // int H = 8;
    int block_len = ((int)blockIdx.x % Tr + 1) * Br;
    int len = seq_k ;// < block_len ? seq_k : block_len;
    int iters = (len+Bc-1) / Bc;
    int k_offset = ((int)blockIdx.x / Tr) * Kd * seq_k;
    int v_offset = ((int)blockIdx.x / Tr) * D * seq_k;
    int q_offset = (int)blockIdx.x  * Kd * Br;
    int o_offset = (int)blockIdx.x  * D * Br;
    int lse_offset = (int)blockIdx.x * Br;
    int m_offset = (int)blockIdx.x * Br;
    int acco_offset = (int)blockIdx.x * Br * D;


    // int m_new = 0;
    // constexpr int lse_new = Br*sizeof(half); // 256
    // int m_old = lse_new+Br*sizeof(half);
    constexpr int p = 0;

    constexpr int Nwarps = Nthreads/32;
    static_assert(Kd%(BlockKSmem)==0,"Kd%(BlockKSmem)!=0");
    static_assert(Bc%(BlockKSmem2)==0,"Bc%(BlockKSmem2)!=0");
    // Gmem copy atom
    static_assert(BlockKSmem2%(Nthreads/(SmemKAtom/8))==0, "BlockKSmem2%(Nthreads/(SmemKAtom/8))!=0");
    static_assert(BlockKSmem%SmemKAtom==0, "BlockKSmem%SmemKAtom!=0");

    using namespace cute;
    TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,Layout<Shape<Int<Nwarps>,_1,_1>>,Layout<Shape<_1,_2,_1>>> tiled_mma;
    Tensor acc_o_fragment = partition_fragment_C(tiled_mma, Shape<Int<Br>,Int<D>>{});

    TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,Layout<Shape<Int<Nwarps>,_1,_1>>,Layout<Shape<_1,_2,_1>>> tiled_mma1;
    Tensor acc_s_fragment = partition_fragment_C(tiled_mma1, Shape<Int<Br>,Int<Bc>>{});// ((_2,_2),_2,_16):((_1,_2),_4,_8)
    half* memeff11_shared = (half *)(shared+p);
    Tensor gQ = make_tensor(make_gmem_ptr(Parameter_0_0_0+q_offset), Shape<Int<Br>,Int<Kd>>{}, make_stride(Int<Kd>{},_1{}));
    Tensor gQ1 = make_tensor(make_gmem_ptr(Parameter_0_0_0+q_offset), Shape<Int<Br>,Int<BlockKSmem>>{}, make_stride(Int<Kd>{},_1{}));
    Tensor gK = make_tensor(make_gmem_ptr(Parameter_1_0_0+k_offset), Shape<Int<Bc>,Int<Kd>>{}, make_stride(Int<Kd>{},_1{}));
    Tensor gK1 = make_tensor(make_gmem_ptr(Parameter_1_0_0+k_offset), Shape<Int<Bc>,Int<BlockKSmem>>{}, make_stride(Int<Kd>{},_1{}));
    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<SmemKAtom>>,
                           Stride<Int<SmemKAtom>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<Br>, Int<Kd>>{}));
    using SmemLayoutQ1 = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<Br>, Int<BlockKSmem>>{}));
    Tensor sQ = make_tensor(make_smem_ptr((half*)(memeff11_shared)), SmemLayoutQ{});
    Tensor sQ1 = make_tensor(make_smem_ptr((half*)(memeff11_shared)), SmemLayoutQ1{});
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<Bc>, Int<Kd>>{}));
    using SmemLayoutK1 = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<Bc>, Int<BlockKSmem>>{}));
    Tensor sK = make_tensor(sQ.data()+size(sQ), SmemLayoutK{});
    Tensor sK1 = make_tensor(sQ1.data()+num_stages_qk*size(sQ1), SmemLayoutK1{});
    using GmemCopyLayoutAtom = Layout<Shape <Int<Nthreads / (SmemKAtom/8)>, Int<SmemKAtom/8>>,
                                  Stride<Int<SmemKAtom/8>, _1>>;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
                        GmemCopyLayoutAtom{},
                        Layout<Shape<_1, _8>>{})); 
    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(threadIdx.x);
    Tensor gQ_partition = gmem_thr_copy_QKV.partition_S(gQ);// (128,256) -> (8(ATOM), M, N)  for(i : M ) for(i:N) copygQ(_,i,j)
    Tensor gQ1_partition = gmem_thr_copy_QKV.partition_S(gQ1);
    Tensor sQ_partition = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor sQ1_partition = gmem_thr_copy_QKV.partition_D(sQ1);
    Tensor gK_partition = gmem_thr_copy_QKV.partition_S(gK);
    Tensor gK1_partition = gmem_thr_copy_QKV.partition_S(gK1);
    Tensor sK_partition = gmem_thr_copy_QKV.partition_D(sK);
    Tensor sK1_partition = gmem_thr_copy_QKV.partition_D(sK1);
    auto thr_mma1 = tiled_mma1.get_thread_slice(threadIdx.x);
    Tensor rQ = thr_mma1.partition_fragment_A(sQ);
    Tensor rQ1 = thr_mma1.partition_fragment_A(sQ1);// (8(ATOM),M,K) 
    Tensor rK = thr_mma1.partition_fragment_B(sK);
    Tensor rK1 = thr_mma1.partition_fragment_B(sK1);// (4(ATOM),N,K)  for(k:K) mma(rQ(_,_,k),rK(_,_,k),acc_s_fragment)
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, half>;
    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma1);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(threadIdx.x);
    Tensor sQ_copypartition = smem_thr_copy_Q.partition_S(sQ); // (ATOM,M,K)
    Tensor sQ1_copypartition = smem_thr_copy_Q.partition_S(sQ1);
    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma1);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(threadIdx.x);
    Tensor sK_copypartition = smem_thr_copy_K.partition_S(sK);
    Tensor sK1_copypartition = smem_thr_copy_K.partition_S(sK1);
    Tensor rQ_copy_view = smem_thr_copy_Q.retile_D(rQ);
    Tensor rQ1_copy_view = smem_thr_copy_Q.retile_D(rQ1); // (ATOM,M,K)
    Tensor rK_copy_view = smem_thr_copy_K.retile_D(rK);
    Tensor rK1_copy_view = smem_thr_copy_K.retile_D(rK1);

    Tensor m_new_fragment = make_tensor<float>(Shape<Int<2*size<1>(acc_s_fragment)>>{});
    // Tensor m_old_fragment = make_fragment_like(m_new_fragment);
    Tensor lse_new_fragment = make_fragment_like(m_new_fragment);

    Tensor gV = make_tensor(make_gmem_ptr(Parameter_2_0_0+v_offset), Shape<Int<Bc>,Int<D>>{}, make_stride(Int<D>{},_1{}));
    // (32,256)
    Tensor gV1 = make_tensor(make_gmem_ptr(Parameter_2_0_0+v_offset), Shape<Int<BlockKSmem2>,Int<D>>{}, make_stride(Int<D>{},_1{}));
    Tensor gV_partition = gmem_thr_copy_QKV.partition_S(gV);
    Tensor gV1_partition = gmem_thr_copy_QKV.partition_S(gV1);
    // This has to be kBlockN and not 8, layout is: KSmem -> KblockN -> Kd(to be consistant with SmemLayoutV)
    using SmemLayoutAtomVtransposedNoSwizzle = Layout<Shape<Int<SmemKAtom>, Int<Bc>>,
                                                      Stride<_1, Int<SmemKAtom>>>;
    using SmemLayoutAtomV1transposedNoSwizzle = Layout<Shape<Int<SmemKAtom>, Int<BlockKSmem2>>,
                                                      Stride<_1, Int<SmemKAtom>>>;
    using SmemLayoutAtomVtransposed = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{}, SmemLayoutAtomVtransposedNoSwizzle{}));
    using SmemLayoutAtomV1transposed = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{}, SmemLayoutAtomV1transposedNoSwizzle{}));
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<Bc>, Int<D>>{}));
    using SmemLayoutV1 = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<BlockKSmem2>, Int<D>>{}));
    using SmemLayoutVtransposed = decltype(tile_to_shape(
        SmemLayoutAtomVtransposed{},
        Shape<Int<D>, Int<Bc>>{}));
    using SmemLayoutV1transposed = decltype(tile_to_shape(
        SmemLayoutAtomV1transposed{},
        Shape<Int<D>, Int<BlockKSmem2>>{}));
    // Maybe the VtransposeNoSwizzle just needs to have the right shape
    // And the strides don't matter?
    using SmemLayoutVtransposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomVtransposedNoSwizzle{},
        Shape<Int<D>, Int<Bc>>{}));
    using SmemLayoutV1transposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomV1transposedNoSwizzle{},
        Shape<Int<D>, Int<BlockKSmem2>>{}));
    //cannot use sQ SMEM
    Tensor sV = make_tensor(sK1.data()+size(sK1)*num_stages_qk,SmemLayoutV{});
    Tensor sV1 = make_tensor(sK1.data()+size(sK1)*num_stages_qk,SmemLayoutV1{});
    Tensor sVt = make_tensor(sV.data(),SmemLayoutVtransposed{});
    Tensor sV1t = make_tensor(sV1.data(),SmemLayoutV1transposed{});
    Tensor sVtNoSwizzle = make_tensor(sVt.data(),SmemLayoutVtransposedNoSwizzle{});
    Tensor sV1tNoSwizzle = make_tensor(sV1t.data(),SmemLayoutV1transposedNoSwizzle{});
    
    Tensor sV_partition = gmem_thr_copy_QKV.partition_D(sV);
    Tensor sV1_partition = gmem_thr_copy_QKV.partition_D(sV1);
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, half>;
    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(threadIdx.x);
    Tensor sVt_copypartition = smem_thr_copy_V.partition_S(sVt);
    Tensor sV1t_copypartition = smem_thr_copy_V.partition_S(sV1t);
    Tensor rVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    Tensor rVt1 = thr_mma.partition_fragment_B(sV1tNoSwizzle);
    Tensor rVt_copy_view = smem_thr_copy_V.retile_D(rVt);
    Tensor rVt1_copy_view = smem_thr_copy_V.retile_D(rVt1);
    
    CopyAsyncQK_g2s cp_g2s_qk(gmem_tiled_copy_QKV, 
                      gQ1_partition, sQ1_partition, 
                      gK1_partition, sK1_partition,
                      BlockKSmem, size(sQ1),
                      BlockKSmem, size(sK1),num_stages_qk);
    CopyAsyncV_g2s cp_g2s_k(gmem_tiled_copy_QKV, 
                      gK1_partition, sK1_partition, 
                      BlockKSmem, size(sK1),num_stages_qk);
    CopyAsyncV_g2s cp_g2s_v(gmem_tiled_copy_QKV, 
                      gV1_partition, sV1_partition, 
                      BlockKSmem2*D, size(sV1),num_stages_v);
    MatmulQK_s2r matmul_qk_s2r(smem_tiled_copy_Q, sQ1_copypartition, rQ1_copy_view, 
                               smem_tiled_copy_K, sK1_copypartition, rK1_copy_view, 
                               tiled_mma1, rQ1, rK1, acc_s_fragment, 
                               size(sQ1), size(sK1),num_stages_qk);
    MatmulV_s2r matmul_v_s2r(smem_tiled_copy_V, sV1t_copypartition, rVt1_copy_view, 
                              tiled_mma, /*rP_Aregs,*/ rVt1, acc_o_fragment, 
                              size(sV1),num_stages_v);

    cp_g2s_qk.prologue();

    cute::fill(lse_new_fragment, 0.0f);
    cute::fill(m_new_fragment, -INFINITY);
    clear(acc_o_fragment);

    // #pragma unroll
    for(int i = 0;i< iters-(unrollLastIter?1:0);i++){

    // q, k -> qk
    clear(acc_s_fragment);
    #pragma unroll
    for(int ax0 = 0;ax0 < Kd/BlockKSmem-1;ax0++){
       cp_async_wait_flash<0>();
      __syncthreads();
      cp_g2s_qk.body();
      matmul_qk_s2r.body();
    }
     cp_async_wait_flash<0>();
    __syncthreads();
    cp_g2s_v.prologue();
    matmul_qk_s2r.epilogue();

    Tensor scores = make_tensor(acc_s_fragment.data(),convert_layout_scores(acc_s_fragment.layout()));// ((2,2)(Atom),M,N) -> ((2,M),(2,N))
    // scores * softmax_scale
    /*#pragma unroll
    for(int ax0 = 0;ax0 < size<0>(scores); ax0++){
      #pragma unroll
      for(int ax1 = 0;ax1 < size<1>(scores); ax1++){
        scores(ax0,ax1) *= softmax_scale;
      }
    }*/
    Tensor m_old_fragment = make_fragment_like(m_new_fragment);
    cute::copy(m_new_fragment, m_old_fragment);
    // m_new
    Tensor scores_max = make_fragment_like(m_new_fragment);
    reduce_max<true>(scores, scores_max);
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(m_new_fragment);ax0++){
      m_new_fragment(ax0) = max(m_new_fragment(ax0), scores_max(ax0)); // lse_new
    }
    // acc_o
    Tensor acc_o_rowcol = make_tensor(acc_o_fragment.data(),  convert_layout_scores(acc_o_fragment.layout()));
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(acc_o_rowcol);ax0++){
        float scale = exp((m_old_fragment(ax0)-m_new_fragment(ax0))*softmax_scale);
        lse_new_fragment(ax0) = lse_new_fragment(ax0) * scale;
        #pragma unroll
        for(int ax1 = 0;ax1 < size<1>(acc_o_rowcol);ax1++){
            acc_o_rowcol(ax0,ax1) *= scale;
        }
    }
    // p = exp(qk-m_new)
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(scores); ax0++){
      float m_scaled = m_new_fragment(ax0)*softmax_scale;
      #pragma unroll
      for(int ax1 = 0;ax1 < size<1>(scores); ax1++){
        scores(ax0,ax1) = exp(scores(ax0,ax1)*softmax_scale - m_scaled);
      }
    }
    // lse
    Tensor scores_sum = make_fragment_like(lse_new_fragment);
    reduce_sum(scores, scores_sum);
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(lse_new_fragment);ax0++){
      lse_new_fragment(ax0) = lse_new_fragment(ax0) + scores_sum(ax0);// m_new_fragment(ax0) + log(exp(lse_new_fragment(ax0)-m_new_fragment(ax0))+scores_sum(ax0));
    }

    cutlass::NumericArrayConverter<half, float, decltype(size(scores))::value> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float,decltype(size(scores))::value> *>(scores.data()));
    Tensor rP = make_tensor(make_rmem_ptr<half>(&frag), scores.layout());
    Tensor rP_Aregs = make_tensor(rP.data(), convert_layout_rowcol_Aregs(rP.layout())); // ((2,M),(2,N)) -> ((2,2,2),M,N/2)

    #pragma unroll
    for(int ax0 = 0;ax0 < Bc/BlockKSmem2-1;ax0++){
       cp_async_wait_flash<0>();
      __syncthreads();
      cp_g2s_v.body();
      matmul_v_s2r.body(rP_Aregs);
    }

     cp_async_wait_flash<0>();
    __syncthreads();
    if(i < iters-1){
      gK1_partition.data() = gK1_partition.data() + (-Kd) + Bc*Kd;
      if(load_q_once){
        cp_g2s_k.prologue();
      }else{
        gQ1_partition.data() = gQ1_partition.data() + (-Kd);
        cp_g2s_qk.prologue();
      }
    }
    matmul_v_s2r.epilogue(rP_Aregs);

    }
    if(unrollLastIter)
    {
      // q, k -> qk
    clear(acc_s_fragment);
    #pragma unroll
    for(int ax0 = 0;ax0 < Kd/BlockKSmem-1;ax0++){
       cp_async_wait_flash<0>();
      __syncthreads();
      cp_g2s_qk.body();
      matmul_qk_s2r.body();
    }
     cp_async_wait_flash<0>();
    __syncthreads();
    cp_g2s_v.prologue();
    matmul_qk_s2r.epilogue();

    Tensor scores = make_tensor(acc_s_fragment.data(),convert_layout_scores(acc_s_fragment.layout()));// ((2,2)(Atom),M,N) -> ((2,M),(2,N))
    // scores * softmax_scale
    /*#pragma unroll
    for(int ax0 = 0;ax0 < size<0>(scores); ax0++){
      #pragma unroll
      for(int ax1 = 0;ax1 < size<1>(scores); ax1++){
        scores(ax0,ax1) *= softmax_scale;
      }
    }*/
    Tensor m_old_fragment = make_fragment_like(m_new_fragment);
    cute::copy(m_new_fragment, m_old_fragment);
    // m_new
    Tensor scores_max = make_fragment_like(m_new_fragment);
    reduce_max<true>(scores, scores_max);
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(m_new_fragment);ax0++){
      m_new_fragment(ax0) = max(m_new_fragment(ax0), scores_max(ax0));
    }
    // acc_o
    Tensor acc_o_rowcol = make_tensor(acc_o_fragment.data(),  convert_layout_scores(acc_o_fragment.layout()));
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(acc_o_rowcol);ax0++){
        float scale = exp((m_old_fragment(ax0)-m_new_fragment(ax0))*softmax_scale);
        lse_new_fragment(ax0) = lse_new_fragment(ax0) * scale;
        #pragma unroll
        for(int ax1 = 0;ax1 < size<1>(acc_o_rowcol);ax1++){
            acc_o_rowcol(ax0,ax1) *= scale;
        }
    }
    // p = exp(qk-m_new)
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(scores); ax0++){
      float m_scaled = m_new_fragment(ax0)*softmax_scale;
      #pragma unroll
      for(int ax1 = 0;ax1 < size<1>(scores); ax1++){
        scores(ax0,ax1) = exp(scores(ax0,ax1)*softmax_scale - m_scaled);
      }
    }
    // lse
    Tensor scores_sum = make_fragment_like(lse_new_fragment);
    reduce_sum(scores, scores_sum);
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(lse_new_fragment);ax0++){
      lse_new_fragment(ax0) = lse_new_fragment(ax0)+scores_sum(ax0);// m_new_fragment(ax0) + log(exp(lse_new_fragment(ax0)-m_new_fragment(ax0))+scores_sum(ax0));
    }

    cutlass::NumericArrayConverter<half, float, decltype(size(scores))::value> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float,decltype(size(scores))::value> *>(scores.data()));
    Tensor rP = make_tensor(make_rmem_ptr<half>(&frag), scores.layout());
    Tensor rP_Aregs = make_tensor(rP.data(), convert_layout_rowcol_Aregs(rP.layout())); // ((2,M),(2,N)) -> ((2,2,2),M,N/2)

    #pragma unroll
    for(int ax0 = 0;ax0 < Bc/BlockKSmem2-1;ax0++){
       cp_async_wait_flash<0>();
      __syncthreads();
      cp_g2s_v.body();
      matmul_v_s2r.body(rP_Aregs);
    }

     cp_async_wait_flash<0>();
    __syncthreads();
    matmul_v_s2r.epilogue(rP_Aregs);

    }

    // out
    Tensor acc_o_rowcol = make_tensor(acc_o_fragment.data(),  convert_layout_scores(acc_o_fragment.layout()));
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(acc_o_rowcol);ax0++){
        float scale = 1/lse_new_fragment(ax0);// exp(m_new_fragment(ax0)-lse_new_fragment(ax0));
        lse_new_fragment(ax0) = m_new_fragment(ax0)*softmax_scale + log(lse_new_fragment(ax0));
        #pragma unroll
        for(int ax1 = 0;ax1 < size<1>(acc_o_rowcol);ax1++){
            acc_o_rowcol(ax0,ax1) *= scale;
        }
    }

    cutlass::NumericArrayConverter<half, float, decltype(size(acc_o_fragment))::value> convert_op2;
    auto frag2 = convert_op2(*reinterpret_cast<const cutlass::Array<float,decltype(size(acc_o_fragment))::value> *>(acc_o_fragment.data()));
    Tensor acc_o_f16 = make_tensor(make_rmem_ptr<half>(&frag2), acc_o_fragment.layout());

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<SmemKAtom>>,
                           Stride<Int<SmemKAtom>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<Br>, Int<D>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, half>;
    Tensor sO = make_tensor(make_smem_ptr((half*)(shared)), SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(threadIdx.x);
    Tensor sO_copypartition = smem_thr_copy_O.partition_D(sO);
    Tensor rO_copy_view = smem_thr_copy_O.retile_S(acc_o_f16);
    Tensor gO = make_tensor(make_gmem_ptr(Result_7_0_0+o_offset), Shape<Int<Br>,Int<D>>{}, make_stride(Int<D>{},_1{}));
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, half>{},
                        GmemCopyLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(threadIdx.x);
    Tensor gO_partition = gmem_thr_copy_O.partition_D(gO);
    Tensor sO_partition = gmem_thr_copy_O.partition_S(sO);
    __syncthreads();
    cute::copy(smem_tiled_copy_O, rO_copy_view, sO_copypartition);
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < size<1>(gO_partition); ++m) {
      #pragma unroll
      for (int k=0;k< size<2>(gO_partition);++k){
        cute::copy(gmem_tiled_copy_O, sO_partition(_, m, k), gO_partition(_, m, k));
      }
    }

}
/*
extern "C" int kernel_entry(half* Parameter_0_0_0, half* Parameter_1_0_0, half* Parameter_2_0_0, half* Result_7_0_0, int B, int H, int Seq_k,int Seq_q)
{
  if(shared_mem > 48*1024){
    cudaFuncSetAttribute(flashattn_fwd, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
  }
    flashattn_fwd<<<dim3(B*H*Seq_q/Br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(Parameter_0_0_0,Parameter_1_0_0,Parameter_2_0_0,Result_7_0_0, H,Seq_k,Seq_q);
// name=Result_7_0
// eliminated: Result_half_half_cuda_lib_Result_7_0(0, Identity_17_0_0, Result_7_0_0);
return 0;
}
*/

