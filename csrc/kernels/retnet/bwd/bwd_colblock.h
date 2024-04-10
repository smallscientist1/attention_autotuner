#pragma once
#include <cuda_fp16.h>
#include "kernels/utils/gmem_copy.h"
#include "kernels/utils/matmul.h"
#include "kernels/utils/misc.h"
#include "kernels/utils/reduce.h"
#include "kernels/utils/elementwise.h"

#include <stdexcept>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#include <stdint.h>

/*
// have bugs when Br or Bc is 32, and SmemKAtomV or SmemKAtom is 32, but Idk why
__device__ constexpr int Br = 64;
constexpr int Bc = 64;
constexpr int Kd = 256;
constexpr int D = 256;
// unroll
constexpr bool unrollLastIter = true;
// for q&k splitk
__device__ constexpr int BlockKSmem = 256;
constexpr int num_stages_qk = 1;
constexpr int num_stages_mask = 1;
// dv,dk
constexpr int num_stages_dv = 1;
// ds
constexpr int num_stages_ds = 1;
// dq
constexpr int num_stages_dq = 1;

constexpr int shared_matmulqk = num_stages_qk*(Br)*BlockKSmem*sizeof(half)+num_stages_qk*Bc*BlockKSmem*sizeof(half);
constexpr int shared_mask = num_stages_mask*Br*Bc*sizeof(half);
constexpr int shared_SdO = Br*Bc*sizeof(half)+Br*D*sizeof(half);
constexpr int shared_v = Bc*D*sizeof(half);
constexpr int shared_mem = shared_matmulqk+shared_mask+shared_SdO+shared_v;
constexpr int Nthreads = 256;
// mma s
constexpr int mmawarpsN = 2;
// mma dv
constexpr int mmawarpsN_dv = 4;
// mma dk
constexpr int mmawarpsN_dk = 4;
// mma dq
constexpr int mmawarpsN_dq = 4;
// for sQ,sK swizzle
constexpr int SmemKAtom = 64;
constexpr int kSwizzle = SmemKAtom == 32 ? 2 : 3;
// for sS swizzle
constexpr int SmemKAtomS = 64;
constexpr int kSwizzleS = SmemKAtomS == 32 ? 2 : 3;
// for sO,sdO swizzle
constexpr int SmemKAtomO = 64;
constexpr int kSwizzleO = SmemKAtomO == 32 ? 2 : 3;
// for mask swizzlw
constexpr int SmemKAtomMask = Bc % 64 == 0 ? 64 : 32;
constexpr int kSwizzleMask = SmemKAtomMask == 32 ? 2 : 3;
// for sV swizzle
constexpr int SmemKAtomV = 64;
constexpr int kSwizzleV = SmemKAtomV == 32 ? 2 : 3;

constexpr int shared_mem_convert_dq = Br*Kd*sizeof(half);
*/


template <typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_type(cute::Tensor<Engine, Layout> const &tensor) {
    using namespace cute;
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template<int Kd, int D, int Br, int Bc, int Nthreads, int mmawarpsN, int mmawarpsN_dv, int mmawarpsN_dk, int mmawarpsN_dq, int BlockKSmem = Kd, int num_stages_qk = 1, bool load_q_once = true, int num_stages_mask = 1, int num_stages_dv = 1, int num_stages_ds = 1, int num_stages_dq = 1, int SmemKAtom = 64, int kSwizzle=3, int SmemKAtomS = 64, int kSwizzleS=3, int SmemKAtomO = 64, int kSwizzleO = 3, int SmemKAtomMask=64, int kSwizzleMask=3 , int SmemKAtomV=64, int kSwizzleV=3, bool unrollLastIter=true>
__global__ void __launch_bounds__(Nthreads) ret_bwd_colblock(half* Parameter_0_0_0, half* Parameter_1_0_0, half* Parameter_2_0_0, half* Parameter_3_0_0, half* Result_7_0_0, float* r, half* dq, half* dk, half* dv,float* dqaccum, int H, int seq_k, int seq_q){

  static_assert(num_stages_dq==1 && num_stages_ds==1 && num_stages_dv==1 && num_stages_mask==1, "not implemented delta stage != 1 ");
  
  extern __shared__ char shared[];
  int Tc = seq_k/Bc;
  int Tr = seq_q/Br;
  int start = 0;// ((int)blockIdx.x % Tc) * Bc;
  int iters_start = start/Br;
  int iters_end = seq_q/Br;
  int k_offset = (int)blockIdx.x * Kd * Bc;
  int v_offset = (int)blockIdx.x * D * Bc;
  int mask_offset = ((int)blockIdx.x % (Tc * H)) / Tc * seq_q * seq_k + ((int)blockIdx.x % Tc) * Bc + iters_start * Br * seq_k;
  int q_offset = ((int)blockIdx.x / Tc) * Kd * seq_q + iters_start * Br*Kd;
  int o_offset = ((int)blockIdx.x / Tc) * D * seq_q + iters_start * Br*D;
  int r_offset = ((int)blockIdx.x / Tc) * seq_q + iters_start * Br;

  constexpr int q = 0;
  constexpr int sMask_offset = q + num_stages_qk*Br*BlockKSmem*sizeof(half) + num_stages_qk*Bc*BlockKSmem*sizeof(half);
  constexpr int S = sMask_offset + num_stages_mask*Br*Bc*sizeof(half); // S(q,k)
  constexpr int dO = S + Br*Bc*sizeof(half);
  constexpr int v = dO + Br*D*sizeof(half);
  constexpr int dS = S;
  constexpr int Nwarps = Nthreads/32;

  using namespace cute;
  TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,Layout<Shape<Int<Nwarps/mmawarpsN>,Int<mmawarpsN>,_1>>,Layout<Shape<_1,_2,_1>>> tiled_mma1;
  Tensor acc_s_fragment = partition_fragment_C(tiled_mma1, Shape<Int<Br>,Int<Bc>>{});// ((_2,_2),_2,_16):((_1,_2),_4,_8)
  half* memeff11_shared = (half *)(shared+q);
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

    Tensor gMask = make_tensor(make_gmem_ptr(Parameter_3_0_0+mask_offset), Shape<Int<Br>,Int<Bc>>{}, make_stride(seq_k,_1{}));
    using SmemLayoutAtomMask = decltype(
        composition(Swizzle<kSwizzleMask, 3, 3>{},
                    Layout<Shape<Int<8>, Int<SmemKAtomMask>>,
                           Stride<Int<SmemKAtomMask>, _1>>{}));
    using SmemLayoutMask = decltype(tile_to_shape(
        SmemLayoutAtomMask{},
        Shape<Int<Br>, Int<Bc>>{}));
    Tensor sMask = make_tensor(make_smem_ptr((half*)(shared+sMask_offset)), SmemLayoutMask{});
    using GmemCopyLayoutAtomMask = Layout<Shape <Int<Nthreads / (SmemKAtomMask/8)>, Int<SmemKAtomMask/8>>,
                                  Stride<Int<SmemKAtomMask/8>, _1>>;
    using GmemTiledCopyMask = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
                        GmemCopyLayoutAtomMask{},
                        Layout<Shape<_1, _8>>{})); 
    GmemTiledCopyMask gmem_tiled_copy_Mask;
    auto gmem_thr_copy_Mask = gmem_tiled_copy_Mask.get_thread_slice(threadIdx.x);
    Tensor gMask_partition = gmem_thr_copy_Mask.partition_S(gMask);
    Tensor sMask_partition = gmem_thr_copy_Mask.partition_D(sMask);
    using SmemCopyAtomMask = Copy_Atom<DefaultCopy, half>;
    auto smem_tiled_copy_mask = make_tiled_copy_C(SmemCopyAtomMask{}, tiled_mma1);
    auto smem_thr_copy_mask = smem_tiled_copy_mask.get_thread_slice(threadIdx.x);
    Tensor rMask = partition_fragment_C(tiled_mma1, Shape<Int<Br>,Int<Bc>>{});
    Tensor rMask_copy_view = smem_thr_copy_mask.retile_D(rMask);
    Tensor sMask_copypartition = smem_thr_copy_mask.partition_S(sMask);

    Tensor gR = make_tensor(make_gmem_ptr(r+r_offset), Shape<Int<Br>>{}, make_stride(_1{}));
    Tensor r_fragment = make_tensor<float>(Shape<Int<2*size<1>(acc_s_fragment)>>{});
    Tensor caccs = make_identity_tensor(Shape<Int<Br>,Int<Bc>>{});
    Tensor caccs_partition1 = thr_mma1.partition_C(caccs);
    static_assert(decltype(size<0>(caccs_partition1))::value == 4); // sm80 mma
    Tensor caccs_partition1_row = logical_divide(caccs_partition1, Shape<Int<2>>{})(make_coord(0,_),_,0);
    
    using SmemLayoutAtomS = decltype(
        composition(Swizzle<kSwizzleS, 3, 3>{},
                    Layout<Shape<_8, Int<SmemKAtomS>>,
                           Stride<Int<SmemKAtomS>, _1>>{}));
    using SmemLayoutS = decltype(tile_to_shape(
        SmemLayoutAtomS{},
        Shape<Int<Br>, Int<Bc>>{}));
    using SmemCopyAtomS = Copy_Atom<DefaultCopy, half>;
    auto smem_tiled_copy_S = make_tiled_copy_C(SmemCopyAtomS{}, tiled_mma1); // not related to tiled_mma1 accum type
    auto smem_thr_copy_S = smem_tiled_copy_S.get_thread_slice(threadIdx.x);
    Tensor sS = make_tensor(make_smem_ptr((half*)(shared+S)), SmemLayoutS{});
    Tensor sS_copypartition = smem_thr_copy_S.partition_D(sS);
    


    TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,Layout<Shape<Int<Nwarps/mmawarpsN_dv>,Int<mmawarpsN_dv>,_1>>,Layout<Shape<_1,_2,_1>>> tiled_mma_dv;
    Tensor acc_dv_fragment = partition_fragment_C(tiled_mma_dv, Shape<Int<Bc>,Int<D>>{});
    auto thr_mma_dv = tiled_mma_dv.get_thread_slice(threadIdx.x);

    using GmemCopyLayoutAtomdO = Layout<Shape <Int<Nthreads / (SmemKAtomO/8)>, Int<SmemKAtomO/8>>,
                                  Stride<Int<SmemKAtomO/8>, _1>>;
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
                        GmemCopyLayoutAtomdO{},
                        Layout<Shape<_1, _8>>{})); 
    GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(threadIdx.x);
    Tensor gdO = make_tensor(make_gmem_ptr(Result_7_0_0+o_offset), Shape<Int<Br>,Int<D>>{}, make_stride(Int<D>{},_1{}));
    Tensor gdO_partition = gmem_thr_copy_dO.partition_S(gdO);
    // dO(for gmem->smem) and dOt(for smem -> reg) must be the same layout in physics
    using SmemLayoutAtomdO = decltype(
        composition(Swizzle<kSwizzleO, 3, 3>{},
                    Layout<Shape<_8, Int<SmemKAtomO>>,
                           Stride<Int<SmemKAtomO>, _1>>{}));
    using SmemLayoutdO = decltype(tile_to_shape(
        SmemLayoutAtomdO{},
        Shape<Int<Br>, Int<D>>{}));
    using SmemLayoutAtomdOtransposedNoSwizzle = Layout<Shape<Int<SmemKAtomO>, Int<Br>>,
                                                      Stride<_1, Int<SmemKAtomO>>>;
    using SmemLayoutAtomdOtransposed = decltype(
        composition(Swizzle<kSwizzleO, 3, 3>{}, SmemLayoutAtomdOtransposedNoSwizzle{}));
    using SmemLayoutdOtransposed = decltype(tile_to_shape(
        SmemLayoutAtomdOtransposed{},
        Shape<Int<D>, Int<Br>>{}));
    using SmemLayoutdOtransposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomdOtransposedNoSwizzle{},
        Shape<Int<D>, Int<Br>>{}));
    Tensor sdO = make_tensor(make_smem_ptr((half*)(shared+dO)), SmemLayoutdO{});
    Tensor sdOt = make_tensor(sdO.data(), SmemLayoutdOtransposed{});
    Tensor sdO_partition = gmem_thr_copy_dO.partition_D(sdO);
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, half>;
    // using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, half>;
    auto smem_tiled_copy_dO = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma_dv);
    auto smem_thr_copy_dO = smem_tiled_copy_dO.get_thread_slice(threadIdx.x);
    Tensor sdOt_copypartition = smem_thr_copy_dO.partition_S(sdOt);
    Tensor sdOtNoSwizzle = make_tensor(sdO.data(), SmemLayoutdOtransposedNoSwizzle{});
    Tensor rdOt = thr_mma_dv.partition_fragment_B(sdOtNoSwizzle);
    Tensor rdOt_copy_view = smem_thr_copy_dO.retile_D(rdOt);

    using SmemLayoutAtomStransposedNoSwizzle = Layout<Shape<Int<SmemKAtomS>, Int<Br>>,
                                                      Stride<_1, Int<SmemKAtomS>>>;
    using SmemLayoutAtomStransposed = decltype(
        composition(Swizzle<kSwizzleS, 3, 3>{}, SmemLayoutAtomStransposedNoSwizzle{}));
    using SmemLayoutStransposed = decltype(tile_to_shape(
        SmemLayoutAtomStransposed{},
        Shape<Int<Bc>, Int<Br>>{}));
    using SmemLayoutStransposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomStransposedNoSwizzle{},
        Shape<Int<Bc>, Int<Br>>{}));
    Tensor sSt = make_tensor(sS.data(), SmemLayoutStransposed{});
    auto smem_tiled_copy_Stransposed = make_tiled_copy_A(SmemCopyAtomTransposed{}, tiled_mma_dv);
    auto smem_thr_copy_Stransposed = smem_tiled_copy_Stransposed.get_thread_slice(threadIdx.x);
    Tensor sSt_copypartition = smem_thr_copy_Stransposed.partition_S(sSt);
    Tensor sStNoSwizzle = make_tensor(sS.data(), SmemLayoutStransposedNoSwizzle{});
    Tensor rSt = thr_mma_dv.partition_fragment_A(sStNoSwizzle);
    Tensor rSt_copy_view = smem_thr_copy_Stransposed.retile_D(rSt);



    // TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,Layout<Shape<Int<Nwarps/2>,_2,_1>>,Layout<Shape<_1,_2,_1>>> tiled_mma_ds;
    Tensor acc_ds_fragment = partition_fragment_C(tiled_mma1, Shape<Int<Br>,Int<Bc>>{});
    // auto thr_mma_ds = tiled_mma_ds.get_thread_slice(threadIdx.x);
    auto smem_tiled_copy_dOA = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma1);
    auto smem_thr_copy_dOA = smem_tiled_copy_dOA.get_thread_slice(threadIdx.x);
    Tensor sdO_copypartition = smem_thr_copy_dOA.partition_S(sdO);
    Tensor rdO  = thr_mma1.partition_fragment_A(sdO);
    Tensor rdO_copy_view = smem_thr_copy_dOA.retile_D(rdO);

    using GmemCopyLayoutAtomV = Layout<Shape <Int<Nthreads / (SmemKAtomV/8)>, Int<SmemKAtomV/8>>,
                                  Stride<Int<SmemKAtomV/8>, _1>>;
    using GmemTiledCopyV = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
                        GmemCopyLayoutAtomV{},
                        Layout<Shape<_1, _8>>{})); 
    GmemTiledCopyV gmem_tiled_copy_V;
    auto gmem_thr_copy_V = gmem_tiled_copy_V.get_thread_slice(threadIdx.x);
    Tensor gV = make_tensor(make_gmem_ptr(Parameter_2_0_0+v_offset), Shape<Int<Bc>,Int<D>>{}, make_stride(Int<D>{},_1{}));
    Tensor gV_partition = gmem_thr_copy_V.partition_S(gV);
    using SmemLayoutAtomV = decltype(
        composition(Swizzle<kSwizzleV, 3, 3>{},
                    Layout<Shape<_8, Int<SmemKAtomV>>,
                           Stride<Int<SmemKAtomV>, _1>>{}));
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<Bc>, Int<D>>{}));
    Tensor sV = make_tensor(make_smem_ptr((half*)(shared+v)), SmemLayoutV{});
    Tensor sV_partition = gmem_thr_copy_V.partition_D(sV);
    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(threadIdx.x);
    Tensor sV_copypartition = smem_thr_copy_V.partition_S(sV);
    Tensor rV = thr_mma1.partition_fragment_B(sV);// Why here can swizzle?
    Tensor rV_copy_view = smem_thr_copy_V.retile_D(rV);

    Tensor sdS = make_tensor(make_smem_ptr((half*)(shared+dS)), SmemLayoutS{});
    Tensor sdS_copypartition = smem_thr_copy_S.partition_D(sdS);


    TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,Layout<Shape<Int<Nwarps/mmawarpsN_dk>,Int<mmawarpsN_dk>,_1>>,Layout<Shape<_1,_2,_1>>> tiled_mma_dk;
    auto thr_mma_dk = tiled_mma_dk.get_thread_slice(threadIdx.x);
    Tensor acc_dk_fragment = partition_fragment_C(tiled_mma_dk, Shape<Int<Bc>,Int<Kd>>{});
    auto smem_tiled_copy_dStransposedA = make_tiled_copy_A(SmemCopyAtomTransposed{}, tiled_mma_dk);
    auto smem_thr_copy_dStransposedA = smem_tiled_copy_dStransposedA.get_thread_slice(threadIdx.x);
    Tensor sdSt = make_tensor(sdS.data(), SmemLayoutStransposed{});
    Tensor sdSt_copypartition = smem_thr_copy_dStransposedA.partition_S(sdSt);
    Tensor sdStNoSwizzle = make_tensor(sdS.data(), SmemLayoutStransposedNoSwizzle{});
    Tensor rdSt = thr_mma_dk.partition_fragment_A(sdStNoSwizzle);
    Tensor rdSt_copy_view = smem_thr_copy_dStransposedA.retile_D(rdSt);

    using SmemLayoutAtomQtransposedNoSwizzle = Layout<Shape<Int<SmemKAtom>, Int<Br>>,
                                                      Stride<_1, Int<SmemKAtom>>>;
    using SmemLayoutAtomQtransposed = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{}, SmemLayoutAtomQtransposedNoSwizzle{}));
    using SmemLayoutQtransposed = decltype(tile_to_shape(
        SmemLayoutAtomQtransposed{},
        Shape<Int<Kd>, Int<Br>>{}));
    using SmemLayoutQtransposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomQtransposedNoSwizzle{},
        Shape<Int<Kd>, Int<Br>>{}));
    Tensor sQt = make_tensor(sQ.data(), SmemLayoutQtransposed{});
    auto smem_tiled_copy_Qtransposed = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma_dk);
    auto smem_thr_copy_Qtransposed = smem_tiled_copy_Qtransposed.get_thread_slice(threadIdx.x);
    Tensor sQt_copypartition = smem_thr_copy_Qtransposed.partition_S(sQt);
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), SmemLayoutQtransposedNoSwizzle{});
    Tensor rQt = thr_mma_dk.partition_fragment_B(sQtNoSwizzle);
    Tensor rQt_copy_view = smem_thr_copy_Qtransposed.retile_D(rQt);



    TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,Layout<Shape<Int<Nwarps/mmawarpsN_dq>,Int<mmawarpsN_dq>,_1>>,Layout<Shape<_1,_2,_1>>> tiled_mma_dq;
    Tensor dq_fragment = partition_fragment_C(tiled_mma_dq, Shape<Int<Br>,Int<Kd>>{});
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(threadIdx.x);
    auto smem_tiled_copy_dSA = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_dq);
    auto smem_thr_copy_dSA = smem_tiled_copy_dSA.get_thread_slice(threadIdx.x);
    Tensor sdS_copypartitionA = smem_thr_copy_dSA.partition_S(sdS);
    Tensor rdS  = thr_mma_dq.partition_fragment_A(sdS);
    Tensor rdS_copy_view = smem_thr_copy_dSA.retile_D(rdS);

    using SmemLayoutAtomKtransposedNoSwizzle = Layout<Shape<Int<SmemKAtom>, Int<Bc>>,
                                                      Stride<_1, Int<SmemKAtom>>>;
    using SmemLayoutAtomKtransposed = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{}, SmemLayoutAtomKtransposedNoSwizzle{}));
    using SmemLayoutKtransposed = decltype(tile_to_shape(
        SmemLayoutAtomKtransposed{},
        Shape<Int<Kd>, Int<Bc>>{}));
    using SmemLayoutKtransposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomKtransposedNoSwizzle{},
        Shape<Int<Kd>, Int<Bc>>{}));
    Tensor sKt = make_tensor(sK.data(), SmemLayoutKtransposed{});
    auto smem_tiled_copy_Ktransposed = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma_dq);
    auto smem_thr_copy_Ktransposed = smem_tiled_copy_Ktransposed.get_thread_slice(threadIdx.x);
    Tensor sKt_copypartition = smem_thr_copy_Ktransposed.partition_S(sKt);
    Tensor sKtNoSwizzle = make_tensor(sK.data(), SmemLayoutKtransposedNoSwizzle{});
    Tensor rKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);
    Tensor rKt_copy_view = smem_thr_copy_Ktransposed.retile_D(rKt);

    using GmemTiledCopyFloatAtomicadd = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, float>{},
                        Layout<Shape<Int<Nthreads/32>, _32>,
                                Stride<_32,_1>>{},
                        Layout<Shape<_1, _1>>{}));
    GmemTiledCopyFloatAtomicadd gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(threadIdx.x);
    Tensor gdQaccum = make_tensor(make_gmem_ptr(dqaccum+q_offset), Shape<Int<Br>,Int<Kd>>{}, make_stride(Int<Kd>{},_1{}));
    Tensor gdQaccum_partition = gmem_thr_copy_dQaccum.partition_D(gdQaccum);



    CopyAsyncQK_g2s cp_g2s_qk(gmem_tiled_copy_QKV, 
                      gQ1_partition, sQ1_partition, 
                      gK1_partition, sK1_partition,
                      BlockKSmem, size(sQ1),
                      BlockKSmem, size(sK1),num_stages_qk);
    CopyAsyncV_g2s cp_g2s_q(gmem_tiled_copy_QKV, 
                      gQ1_partition, sQ1_partition, 
                      BlockKSmem, size(sQ1),num_stages_qk);
    CopyAsyncV_g2s cp_g2s_mask(gmem_tiled_copy_Mask, 
                      gMask_partition, sMask_partition, 
                      Br*seq_k, size(sMask),num_stages_mask);    
    MatmulQK_s2r matmul_qk_s2r(smem_tiled_copy_Q, sQ1_copypartition, rQ1_copy_view, 
                               smem_tiled_copy_K, sK1_copypartition, rK1_copy_view, 
                               tiled_mma1, rQ1, rK1, acc_s_fragment, 
                               size(sQ1), size(sK1),num_stages_qk);
    
    CopyAsyncV_g2s cp_g2s_do(gmem_tiled_copy_dO, 
                      gdO_partition, sdO_partition, 
                      size(gdO), size(sdO),num_stages_dv);
    MatmulQsharedK_s2r matmul_sdo_s2r( 
                              smem_tiled_copy_Stransposed, sSt_copypartition, rSt_copy_view, 
                              smem_tiled_copy_dO, sdOt_copypartition, rdOt_copy_view,
                              tiled_mma_dv, rSt, rdOt, acc_dv_fragment, 
                              size(sSt), size(sdOt),num_stages_dv);

    CopyAsyncV_g2s cp_g2s_v(gmem_tiled_copy_V, 
                      gV_partition, sV_partition, 
                      size(gV), size(sV),num_stages_ds);
    /*if(threadIdx.x==0 && blockIdx.x==0){
      print(sV.layout());// S<2,3,3> o _0 o (_32,(_32,_16)):(_32,(_1,_1024))
      printf("\n%d\n", int(size(sV)));// 0???要加int()
      printf("\n%d\n",int(size(sQ)));
    }*/
    MatmulQsharedK_s2r matmul_dov_s2r( 
                              smem_tiled_copy_dOA, sdO_copypartition, rdO_copy_view,
                              smem_tiled_copy_V, sV_copypartition, rV_copy_view,
                              tiled_mma1, rdO, rV, acc_ds_fragment, 
                              size(sdO), size(sV),num_stages_ds);

    MatmulQK_s2r matmul_dsq_s2r(smem_tiled_copy_dStransposedA, sdSt_copypartition, rdSt_copy_view, 
                               smem_tiled_copy_Qtransposed, sQt_copypartition, rQt_copy_view, 
                               tiled_mma_dk, rdSt, rQt, acc_dk_fragment, 
                               size(sdSt), size(sQt),num_stages_dv);

    MatmulQK_s2r matmul_dsk_s2r(smem_tiled_copy_dSA, sdS_copypartitionA, rdS_copy_view, 
                               smem_tiled_copy_Ktransposed, sKt_copypartition, rKt_copy_view,
                               tiled_mma_dq, rdS, rKt, dq_fragment,
                                size(sdS), size(sKt),num_stages_dq);

  clear(acc_dk_fragment);
  clear(acc_dv_fragment);

  cp_g2s_qk.prologue();
  cp_g2s_v.prologue();

  for(int i=iters_start;i<iters_end-(unrollLastIter?1:0);i++){
    // load r
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(r_fragment); ax0++){
      const int row = get<0>(caccs_partition1_row(ax0));
      r_fragment(ax0) = gR(row);
    }
    gR.data() = gR.data() + Br;
    // q @ k
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
    cp_g2s_mask.prologue();
    matmul_qk_s2r.epilogue();

    cp_async_wait_flash<0>();
    __syncthreads();
    cp_g2s_do.prologue();

    /*if(threadIdx.x==0 && blockIdx.x==0){
      printf("\nacc_s_fragment\n");
      for(int i=0;i<size(acc_s_fragment);i++){
        printf("%d:%f ", i, acc_s_fragment(i));
      }
    }*/
    /*if(threadIdx.x==0 && blockIdx.x==0){
      printf("\nsMask\n");
      for(int i=0;i<size<0>(sMask);i++){
        for(int j=0;j<size<1>(sMask);j++){
          printf("%d,%d:%f ", i,j, __half2float(sMask(i,j)));
        }
      }
    }*/
    // qk*m
    multiply_mask(smem_tiled_copy_mask, sMask_copypartition, rMask_copy_view, acc_s_fragment, rMask);

    // qkm/r
    Tensor scores = make_tensor(acc_s_fragment.data(),convert_layout_scores(acc_s_fragment.layout()));// ((2,2)(Atom),M,N) -> ((2,M),(2,N))
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(scores); ax0++){
      #pragma unroll
      for(int ax1 = 0;ax1 < size<1>(scores); ax1++){
        scores(ax0,ax1) = scores(ax0,ax1) / r_fragment(ax0);
      }
    }
    /*if(threadIdx.x==0 && blockIdx.x==0){
      printf("\nr_fragment\n");
      for(int i=0;i<size(r_fragment);i++){
        printf("%d:%f ", i, r_fragment(i));
      }
      printf("\nacc_s_fragment\n");
      for(int i=0;i<size(acc_s_fragment);i++){
        printf("%d:%f ", i, acc_s_fragment(i));
      }
      print(acc_s_fragment.layout());
      print("\n");
    }*/
    Tensor rP_f16 = convert_type<half>(acc_s_fragment);
    Tensor rP_f16_copy_view = smem_thr_copy_S.retile_S(rP_f16);
    /*if(threadIdx.x==0 && blockIdx.x==0){
      printf("\nrP_f16\n");
      for(int i=0;i<size(rP_f16);i++){
          printf("%d:%f ", i, __half2float(rP_f16(i)));
      }
    }*/
    cute::copy(smem_tiled_copy_S, rP_f16_copy_view, sS_copypartition);
    // finish copy do
    cp_async_wait_flash<0>();
    __syncthreads();
    // dv = St @ dO
    matmul_sdo_s2r.epilogue();
    __syncthreads(); // syncthreads here because ds & s use the same memory
    /*if(threadIdx.x==0 && blockIdx.x==0){
      printf("\nacc_dv_fragment\n");
      for(int i=0;i<size(acc_dv_fragment);i++){
        printf("%d:%f ", i, acc_dv_fragment(i));
      }
      printf("\nsS\n");
      for(int i=0;i<size<0>(sS);i++){
        for(int j=0;j<size<1>(sS);j++){
          printf("%d,%d:%f ", i,j, __half2float(sS(i,j)));
        }
      }
    }*/
    // second part
    // ds = do @ Vt
    clear(acc_ds_fragment);
    matmul_dov_s2r.epilogue();
    // ds/r
    Tensor dscores = make_tensor(acc_ds_fragment.data(),convert_layout_scores(acc_ds_fragment.layout()));// ((2,2)(Atom),M,N) -> ((2,M),(2,N))
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(dscores); ax0++){
      #pragma unroll
      for(int ax1 = 0;ax1 < size<1>(dscores); ax1++){
        dscores(ax0,ax1) = dscores(ax0,ax1) / r_fragment(ax0);
      }
    }
    // dqk = ...*mask(can reg fuse with previos apply mask?)
    multiply_mask(smem_tiled_copy_mask, sMask_copypartition, rMask_copy_view, acc_ds_fragment, rMask);
    Tensor rdS_f16 = convert_type<half>(acc_ds_fragment);
    Tensor rdS_f16_copy_view = smem_thr_copy_S.retile_S(rdS_f16);
    cute::copy(smem_tiled_copy_S, rdS_f16_copy_view, sdS_copypartition);
    __syncthreads();
    // dk = dst @ q
    matmul_dsq_s2r.epilogue();
    __syncthreads();// for copy q
    /*if(threadIdx.x==0 && blockIdx.x==0){
      printf("\nacc_dk_fragment\n");
      for(int i=0;i<size(acc_dk_fragment);i++){
        printf("%d:%f ", i, acc_dk_fragment(i));
      }
    }*/
    // copy q
    if(i<iters_end-1){
      gQ1_partition.data() = gQ1_partition.data() + (-Kd) + Br*Kd;
      cp_g2s_q.prologue();
    }
    // dq = ds @ k(can reg fuse?)
    clear(dq_fragment);
    matmul_dsk_s2r.epilogue();
    /*if(threadIdx.x==0 && blockIdx.x==0){
      printf("\ndq_fragment\n");
      for(int i=0;i<size(dq_fragment);i++){
        printf("%d:%f ", i, dq_fragment(i));
      }
    }*/
    #pragma unroll
    for(int ax0 = 0;ax0 < size(dq_fragment); ax0++){
      atomicAdd(&gdQaccum_partition(ax0), dq_fragment(ax0));
    }
    /*if(threadIdx.x==0 && blockIdx.x==0){
      printf("\ngdQaccum\n");
      for(int i=0;i<size<0>(gdQaccum);i++){
        for(int j=0;j<size<1>(gdQaccum);j++){
          printf("%d,%d:%f ", i,j, gdQaccum(i,j));
        }
      }
    }*/
    gdQaccum_partition.data() = gdQaccum_partition.data() + size(gdQaccum);
  }
  if(unrollLastIter){
    // load r
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(r_fragment); ax0++){
      const int row = get<0>(caccs_partition1_row(ax0));
      r_fragment(ax0) = gR(row);
    }
    gR.data() = gR.data() + Br;
    // q @ k
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
    cp_g2s_mask.prologue();
    matmul_qk_s2r.epilogue();

    cp_async_wait_flash<0>();
    __syncthreads();
    cp_g2s_do.prologue();

    // qk*m
    multiply_mask(smem_tiled_copy_mask, sMask_copypartition, rMask_copy_view, acc_s_fragment, rMask);

    // qkm/r
    Tensor scores = make_tensor(acc_s_fragment.data(),convert_layout_scores(acc_s_fragment.layout()));// ((2,2)(Atom),M,N) -> ((2,M),(2,N))
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(scores); ax0++){
      #pragma unroll
      for(int ax1 = 0;ax1 < size<1>(scores); ax1++){
        scores(ax0,ax1) = scores(ax0,ax1) / r_fragment(ax0);
      }
    }
    Tensor rP_f16 = convert_type<half>(acc_s_fragment);
    Tensor rP_f16_copy_view = smem_thr_copy_S.retile_S(rP_f16);
    cute::copy(smem_tiled_copy_S, rP_f16_copy_view, sS_copypartition);
    // finish copy do
    cp_async_wait_flash<0>();
    __syncthreads();
    // dv = St @ dO
    matmul_sdo_s2r.epilogue();
    __syncthreads(); // syncthreads here because ds & s use the same memory
    /*if(threadIdx.x==0 && blockIdx.x==0){
      printf("\nacc_dv_fragment\n");
      for(int i=0;i<size(acc_dv_fragment);i++){
        printf("%d:%f ", i, acc_dv_fragment(i));
      }
      printf("\nsS\n");
      for(int i=0;i<size<0>(sS);i++){
        for(int j=0;j<size<1>(sS);j++){
          printf("%d,%d:%f ", i,j, __half2float(sS(i,j)));
        }
      }
    }*/
    // second part
    // ds = do @ Vt
    clear(acc_ds_fragment);
    matmul_dov_s2r.epilogue();
    // ds/r
    Tensor dscores = make_tensor(acc_ds_fragment.data(),convert_layout_scores(acc_ds_fragment.layout()));// ((2,2)(Atom),M,N) -> ((2,M),(2,N))
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(dscores); ax0++){
      #pragma unroll
      for(int ax1 = 0;ax1 < size<1>(dscores); ax1++){
        dscores(ax0,ax1) = dscores(ax0,ax1) / r_fragment(ax0);
      }
    }
    // dqk = ...*mask(can reg fuse with previos apply mask?)
    multiply_mask(smem_tiled_copy_mask, sMask_copypartition, rMask_copy_view, acc_ds_fragment, rMask);
    Tensor rdS_f16 = convert_type<half>(acc_ds_fragment);
    Tensor rdS_f16_copy_view = smem_thr_copy_S.retile_S(rdS_f16);
    cute::copy(smem_tiled_copy_S, rdS_f16_copy_view, sdS_copypartition);
    __syncthreads();
    // dk = dst @ q
    matmul_dsq_s2r.epilogue();
    // dq = ds @ k(can reg fuse?)
    clear(dq_fragment);
    matmul_dsk_s2r.epilogue();
    #pragma unroll
    for(int ax0 = 0;ax0 < size(dq_fragment); ax0++){
      atomicAdd(&gdQaccum_partition(ax0), dq_fragment(ax0));
    }
    gdQaccum_partition.data() = gdQaccum_partition.data() + size(gdQaccum);
  }

  // store dk dv
  Tensor rdk_f16 = convert_type<half>(acc_dk_fragment);
  Tensor rdv_f16 = convert_type<half>(acc_dv_fragment);

  using SmemCopyAtomdk = Copy_Atom<DefaultCopy, half>;
  Tensor sdk = make_tensor(make_smem_ptr((half*)(shared+0)), SmemLayoutK{});
  Tensor sdv = make_tensor(sdk.data()+size(sdk), SmemLayoutV{});
  auto smem_tiled_copy_dk = make_tiled_copy_C(SmemCopyAtomdk{}, tiled_mma_dk);
  auto smem_thr_copy_dk = smem_tiled_copy_dk.get_thread_slice(threadIdx.x);
  Tensor sdk_copypartition = smem_thr_copy_dk.partition_D(sdk);
  Tensor rdk_f16_copy_view = smem_thr_copy_dk.retile_S(rdk_f16);
  auto smem_tiled_copy_dv = make_tiled_copy_C(SmemCopyAtomdk{}, tiled_mma_dv);
  auto smem_thr_copy_dv = smem_tiled_copy_dv.get_thread_slice(threadIdx.x);
  Tensor sdv_copypartition = smem_thr_copy_dv.partition_D(sdv);
  Tensor rdv_f16_copy_view = smem_thr_copy_dv.retile_S(rdv_f16);
  Tensor gdK = make_tensor(make_gmem_ptr(dk+k_offset), Shape<Int<Bc>,Int<Kd>>{}, make_stride(Int<Kd>{},_1{}));
  Tensor gdV = make_tensor(make_gmem_ptr(dv+v_offset), Shape<Int<Bc>,Int<D>>{}, make_stride(Int<D>{},_1{}));
  using GmemTiledCopydK = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, half>{},
                        GmemCopyLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));
  GmemTiledCopydK gmem_tiled_copy_dK;
  auto gmem_thr_copy_dK = gmem_tiled_copy_dK.get_thread_slice(threadIdx.x);
  Tensor gdK_partition = gmem_thr_copy_dK.partition_D(gdK);
  Tensor sdK_partition = gmem_thr_copy_dK.partition_S(sdk);
  using GmemTIledCopydV = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, half>{},
                        GmemCopyLayoutAtomV{},
                        Layout<Shape<_1, _8>>{}));
  GmemTIledCopydV gmem_tiled_copy_dV;
  auto gmem_thr_copy_dV = gmem_tiled_copy_dV.get_thread_slice(threadIdx.x);
  Tensor gdV_partition = gmem_thr_copy_dV.partition_D(gdV);
  Tensor sdV_partition = gmem_thr_copy_dV.partition_S(sdv);

  __syncthreads();
  cute::copy(smem_tiled_copy_dk, rdk_f16_copy_view, sdk_copypartition);
  cute::copy(smem_tiled_copy_dv, rdv_f16_copy_view, sdv_copypartition);
  __syncthreads();//one syncthread for dkdv
  #pragma unroll
  for(int ax0 = 0; ax0 < size<1>(gdK_partition); ax0++){
    #pragma unroll
    for(int ax1 = 0; ax1 < size<2>(gdK_partition); ax1++){
      cute::copy(gmem_tiled_copy_dK, sdK_partition(_,ax0,ax1), gdK_partition(_,ax0,ax1));
    }
  }
  #pragma unroll
  for(int ax0 = 0; ax0 < size<1>(gdV_partition); ax0++){
    #pragma unroll
    for(int ax1 = 0; ax1 < size<2>(gdV_partition); ax1++){
      cute::copy(gmem_tiled_copy_dV, sdV_partition(_,ax0,ax1), gdV_partition(_,ax0,ax1));
    }
  }

}

template< int Kd, int Br, int Bc, int Nthreads,int mmawarpsN_dq , int SmemKAtom=64, int kSwizzle=3>
__global__ void convert_dq(half* dq, float* dqaccum, int seq_k, int seq_q){
  extern __shared__ char shared[];

  int Tc = seq_k/Bc;
  int Tr = seq_q/Br;
  int q_offset = (int)blockIdx.x * Kd * Br;
  constexpr int Nwarps = Nthreads/32;

  using namespace cute;
  // convert dQ
  constexpr int sdq_offset = 0;
  Tensor gdQ = make_tensor(make_gmem_ptr(dq+q_offset), Shape<Int<Br>,Int<Kd>>{}, make_stride(Int<Kd>{},_1{}));
  using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<SmemKAtom>>,
                           Stride<Int<SmemKAtom>, _1>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<Br>, Int<Kd>>{}));
  Tensor sdQ = make_tensor(make_smem_ptr((half*)(shared+sdq_offset)), SmemLayoutQ{});
  
  using GmemCopyLayoutAtom = Layout<Shape <Int<Nthreads / (SmemKAtom/8)>, Int<SmemKAtom/8>>,
                                  Stride<Int<SmemKAtom/8>, _1>>;
  using GmemTiledCopyFloatAtomicadd = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, float>{},
                        Layout<Shape<Int<Nthreads/32>, _32>,
                                Stride<_32,_1>>{},
                        Layout<Shape<_1, _1>>{}));
  GmemTiledCopyFloatAtomicadd gmem_tiled_copy_dQaccum;
  auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(threadIdx.x);
  using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, half>{},
                        GmemCopyLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));
  GmemTiledCopydQ gmem_tiled_copy_dQ;
  auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(threadIdx.x);
  Tensor gdQaccum1 = make_tensor(make_gmem_ptr(dqaccum+q_offset), Shape<Int<Br>,Int<Kd>>{}, make_stride(Int<Kd>{},_1{}));
  Tensor gdQaccum1_partition = gmem_thr_copy_dQaccum.partition_S(gdQaccum1);
  
  TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,Layout<Shape<Int<Nwarps/mmawarpsN_dq>,Int<mmawarpsN_dq>,_1>>,Layout<Shape<_1,_2,_1>>> tiled_mma_dq;
  Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<Br>,Int<Kd>>{});
  Tensor rdQaccum1 = make_fragment_like(gdQaccum1_partition);

  using SmemCopyAtomdQ = Copy_Atom<DefaultCopy, half>;
  auto smem_tiled_copy_dQ = make_tiled_copy_C(SmemCopyAtomdQ{}, tiled_mma_dq);
  auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(threadIdx.x);
  Tensor sdQ_copypartition = smem_thr_copy_dQ.partition_D(sdQ);
  
  Tensor gdQ_partition = gmem_thr_copy_dQ.partition_D(gdQ);
  Tensor sdQ_partition = gmem_thr_copy_dQ.partition_S(sdQ);


  cute::copy(gmem_tiled_copy_dQ, gdQaccum1_partition, rdQaccum1);
  #pragma unroll
  for(int i=0;i<size(acc_dq);i++){
    acc_dq(i) = rdQaccum1(i);
  }
  Tensor rdQ = convert_type<half>(acc_dq);
  Tensor rdQ_copy_view = smem_thr_copy_dQ.retile_S(rdQ);
  cute::copy(smem_tiled_copy_dQ, rdQ_copy_view, sdQ_copypartition);
  __syncthreads();
  #pragma unroll
  for (int m = 0; m < size<1>(gdQ_partition); ++m) {
      #pragma unroll
      for (int k=0;k< size<2>(gdQ_partition);++k){
        cute::copy(gmem_tiled_copy_dQ, sdQ_partition(_, m, k), gdQ_partition(_, m, k));
      }
  }
}

/*
extern "C" int kernel_entry(half* Parameter_0_0_0, half* Parameter_1_0_0, half* Parameter_2_0_0, half* Parameter_3_0_0, half* Result_7_0_0,float* r, half* dq, half* dk, half* dv, int B, int H, int Seq_k,int Seq_q)
{
  if(shared_mem > 48*1024){
    cudaFuncSetAttribute(ret_bwd, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
  }
  if(shared_mem_convert_dq > 48*1024){
    cudaFuncSetAttribute(convert_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_convert_dq);
  }
  float* dqaccum;
  CUDA_SAFE_CALL(cudaMalloc((void**)&dqaccum, sizeof(float)*B*H*Seq_q*Kd));
  CUDA_SAFE_CALL(cudaMemset(dqaccum, 0, sizeof(float)*B*H*Seq_q*Kd));
  ret_bwd<<<dim3(B*H*Seq_k/Bc, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(Parameter_0_0_0,Parameter_1_0_0,Parameter_2_0_0,Parameter_3_0_0,Result_7_0_0,r,dq,dk,dv,dqaccum, H,Seq_k,Seq_q);
  // convert_dkv<<<dim3(B*H*Seq_k/Bc, 1, 1), dim3(Nthreads, 1, 1),shared_mem_convert_dkv,0>>>(dk,dv,dkaccum,dvaccum,Seq_k,Seq_q);
  convert_dq<<<dim3(B*H*Seq_q/Br, 1, 1), dim3(Nthreads, 1, 1),shared_mem_convert_dq,0>>>(dq,dqaccum,Seq_k,Seq_q);
  CUDA_SAFE_CALL(cudaFree(dqaccum));
// name=Result_7_0
// eliminated: Result_half_half_cuda_lib_Result_7_0(0, Identity_17_0_0, Result_7_0_0);
return 0;
}
*/
