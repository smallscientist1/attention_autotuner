#pragma once

#include <cuda_fp16.h>
#include <mma.h>

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
constexpr int Br = 64;
constexpr int Bc = 64; //128
constexpr int Kd = 256;
constexpr int D = 448;

constexpr bool unrollLastIter = true;
// for sQ,sK,sP_f16 swizzle
constexpr int SmemKAtom = 64;
constexpr int kSwizzle = SmemKAtom == 32 ? 2 : 3;
// for sV swizzle
constexpr int SmemKAtomV = 64;
constexpr int kSwizzleV = SmemKAtomV == 32 ? 2 : 3;
// for sP swizzle
constexpr int SmemKAtomP = Bc % 64 == 0 ? 64 : 32;
constexpr int kSwizzleP = SmemKAtomP == 32 ? 2 : 3;
// for sP_f16 swizzle
constexpr int SmemKAtomPf16 = 64;
constexpr int kSwizzlePf16 = SmemKAtomPf16 == 32 ? 2 : 3;
// for mask swizzlw
constexpr int SmemKAtomMask = Bc % 64 == 0 ? 64 : 32;
constexpr int kSwizzleMask = SmemKAtomMask == 32 ? 2 : 3;
// for q&k splitk
__device__ constexpr int BlockKSmem = 256; // avoid 'error: identifier "BlockKSmem" is undefined in device code'
constexpr int num_stages_qk = 1;
constexpr int num_stages_mask = 1;
// for V splitk
constexpr int BlockKSmem2 = 64;
constexpr int num_stages_v = 1;
// mma1 N
constexpr int warps_mma1_N = 2;
// mma N
constexpr int warps_mma_N = 4;

constexpr int shared_matmulqkv = num_stages_qk*(Br)*BlockKSmem*sizeof(half)+num_stages_qk*Bc*BlockKSmem*sizeof(half)+num_stages_v*BlockKSmem2*D*sizeof(half);
constexpr int shared_accs = Br*Bc*sizeof(float)+Br*Bc*sizeof(half) + 2*sizeof(float)*Br;
constexpr int shared_mask = num_stages_mask*Br*Bc*sizeof(half);
constexpr int shared_mem = shared_matmulqkv + shared_accs + shared_mask;//(acc_o(p(q,k),v))

constexpr int Nthreads = 256;
*/



template<int Kd, int D, int Br, int Bc, int Nthreads, int BlockKSmem=Kd, int num_stages_qk=1,bool load_q_once=true,int BlockKSmem2=Bc,int num_stages_v=1,int num_stages_mask=1,int warps_mma1_N=2, int warps_mma_N=4,int SmemKAtom=64,int kSwizzle=3,int SmemKAtomV=64, int kSwizzleV=3,int SmemKAtomMask=64,int kSwizzleMask=3,int SmemKAtomP=64,int kSwizzleP=3,int SmemKAtomPf16=64, int kSwizzlePf16=3,bool unrollLastIter=true>
__global__ void __launch_bounds__(Nthreads) ret_fwd_smemfuse(half* Parameter_0_0_0, half* Parameter_1_0_0, half* Parameter_2_0_0, half* Parameter_3_0_0, half* Result_7_0_0, float* r, int H, int seq_k, int seq_q){

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
    int mask_offset = ((int)blockIdx.x%(Tr * H)) * Br * seq_k;
    int o_offset = (int)blockIdx.x  * D * Br;
    int lse_offset = (int)blockIdx.x * Br;

    constexpr int sMask_offset = 0;
    constexpr int r_new = sMask_offset + num_stages_mask * Br * Bc*sizeof(half);
    // constexpr int r_wo_clamp = r_new + Br*sizeof(float); //
    int r_old = r_new+Br*sizeof(float);
    constexpr int p = r_new+2*Br*sizeof(float); // p
    constexpr int p_f16 = p+Br*Bc*sizeof(float); // p_f16
    constexpr int q = p_f16 + Br*Bc*sizeof(half); // q

    constexpr int Nwarps = Nthreads/32;
    static_assert(Kd%(BlockKSmem)==0,"Kd%(BlockKSmem)!=0");
    static_assert(Bc%(BlockKSmem2)==0,"Bc%(BlockKSmem2)!=0");
    // Gmem copy atom
    static_assert(BlockKSmem2%(Nthreads/(SmemKAtomV/8))==0, "gmem load V fail");
    static_assert(Br%(Nthreads/(SmemKAtom/8))==0, "gmem load Q fail");
    static_assert(Bc%(Nthreads/(SmemKAtom/8))==0, "gmem load K fail");
    // Smem Layout
    static_assert(BlockKSmem%SmemKAtom==0, "BlockKSmem%SmemKAtom!=0");
    static_assert(BlockKSmem2%SmemKAtomPf16==0, "BlockKSmem2%SmemKAtomPf16!=0");

    using namespace cute;
    TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,Layout<Shape<Int<Nwarps/warps_mma_N>,Int<warps_mma_N>,_1>>,Layout<Shape<_1,_2,_1>>> tiled_mma; // must be _2 for cute::SM75_U16x8_LDSM_T
    Tensor acc_o_fragment = partition_fragment_C(tiled_mma, Shape<Int<Br>,Int<D>>{});

    TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,Layout<Shape<Int<Nwarps/warps_mma1_N>,Int<warps_mma1_N>,_1>>,Layout<Shape<_1,_2,_1>>> tiled_mma1;
    Tensor acc_s_fragment = partition_fragment_C(tiled_mma1, Shape<Int<Br>,Int<Bc>>{});// ((_2,_2),_2,_16):((_1,_2),_4,_8)
    
    // bankconflict2 when fragmentC->smem because of float
    using SmemLayoutAtomP = decltype(
        composition(Swizzle<kSwizzleP, 3, 3>{},
                    Layout<Shape<_8, Int<SmemKAtomP>>,
                           Stride<Int<SmemKAtomP>, _1>>{}));
    using SmemLayoutP = decltype(tile_to_shape(
        SmemLayoutAtomP{},
        Shape<Int<Br>,Int<Bc>>{}));
    Tensor sP = make_tensor(make_smem_ptr((float*)(shared+p)), SmemLayoutP{});
    using SmemCopyAtomP = Copy_Atom<DefaultCopy, float>;
    auto smem_tiled_copy_P1 = make_tiled_copy_C(SmemCopyAtomP{}, tiled_mma1);
    auto smem_thr_copy_P1 = smem_tiled_copy_P1.get_thread_slice(threadIdx.x);
    Tensor sP_copypartition1 = smem_thr_copy_P1.partition_D(sP);
    Tensor rP_copy_view1 = smem_thr_copy_P1.retile_S(acc_s_fragment);

    using SmemLayoutAtomPf16 = decltype(
        composition(Swizzle<kSwizzlePf16, 3, 3>{},
                    Layout<Shape<_8, Int<SmemKAtomPf16>>,
                           Stride<Int<SmemKAtomPf16>, _1>>{}));
    using SmemLayoutPf16 = decltype(tile_to_shape(
        SmemLayoutAtomPf16{},
        Shape<Int<Br>,Int<Bc>>{}));
    Tensor sP_f16 = make_tensor(make_smem_ptr((half*)(shared+p_f16)), SmemLayoutPf16{});
    // the Layout of sP_f161 is the same as sP_f16 because tilr_to_shape is column major
    using SmemLayoutPf161 = decltype(tile_to_shape(
        SmemLayoutAtomPf16{},
        Shape<Int<Br>,Int<BlockKSmem2>>{}));
    Tensor sP_f161 = make_tensor(make_smem_ptr((half*)(shared+p_f16)), SmemLayoutPf161{});

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

    // only for acc_o = p @ v
    using SmemLayoutsm = Layout<Shape<Int<Br>,Int<D>>,Stride<_1,_0>>;
    // Tensor sr_wo_clamp = make_tensor(make_smem_ptr((float*)(shared+r_wo_clamp)), SmemLayoutsm{});
    Tensor sr_new = make_tensor(make_smem_ptr((float*)(shared+r_new)), SmemLayoutsm{});
    Tensor sr_old = make_tensor(make_smem_ptr((float*)(shared+r_old)), SmemLayoutsm{});
    using SmemCopyAtomFloat = Copy_Atom<DefaultCopy, float>;
    auto smem_tiled_copy_sm = make_tiled_copy_C(SmemCopyAtomFloat{}, tiled_mma);
    auto smem_thr_copy_sm = smem_tiled_copy_sm.get_thread_slice(threadIdx.x);
    Tensor sr_new_copypartition = smem_thr_copy_sm.partition_S(sr_new);
    Tensor sr_old_copypartition = smem_thr_copy_sm.partition_S(sr_old);
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    // Tensor rm = make_tensor<float>(Layout<Shape<Shape<_2,_2>,Int<Br/(1*16)>,Int<D/(8*8)>>,Stride<Stride<_0,_1>,_2,_0>>{});
    // Tensor rm_copy_view = smem_thr_copy_sm.retile_D(rm);


    constexpr int nthreadsPerRow = 8;
    using CopyLayoutFloat = Layout<Shape<Int<Nthreads/nthreadsPerRow>, Int<nthreadsPerRow>>,Stride<Int<nthreadsPerRow>,_1>>;
    auto smem_tiled_copy_P2 = make_tiled_copy(SmemCopyAtomFloat{}, CopyLayoutFloat{}, Layout<Shape<_1,Int<SmemKAtomP/nthreadsPerRow>>>{});
    auto smem_thr_copy_P2 = smem_tiled_copy_P2.get_thread_slice(threadIdx.x);
    Tensor sP_copypartition2 = smem_thr_copy_P2.partition_S(sP);
    Tensor scores = make_tensor<float>(Layout<Shape<Int<Br/(Nthreads/nthreadsPerRow)>,Int<Bc/nthreadsPerRow>>,Stride<Int<Bc/nthreadsPerRow>,_1>>{});
    Tensor scores_cp_view = make_tensor(scores.data(),convert_layout_scores_copyview<SmemKAtomP/nthreadsPerRow>(scores.layout()));
    Tensor scores_copy_view = smem_thr_copy_P2.retile_D(scores_cp_view);
    
    // r_wo_clamp in register, r_new and r_old in smem, r_new store to smem every loop
    Tensor r_wo_clamp_fragment = make_tensor<float>(Shape<Int<size<0>(scores)>>{});
    Tensor r_new_fragment = make_fragment_like(r_wo_clamp_fragment);
    Tensor r_old_fragment = make_fragment_like(r_new_fragment);


    /*using CopyLayoutM = Layout<Shape<Int<(Nthreads/8)>,Int<8>>,Stride<_1,_0>>;
    auto smem_tiled_copy_M = make_tiled_copy(SmemCopyAtomFloat{}, CopyLayoutM{}, Layout<Shape<_1,_1>>{});
    auto smem_thr_copy_M = smem_tiled_copy_M.get_thread_slice(threadIdx.x);
    auto sm_new_copypatition = smem_thr_copy_M.partition_S(sm_new);*/
    using SmemCopyAtomScoresf16 = Copy_Atom<DefaultCopy, half>;
    using CopyLayoutScoresf16 = Layout<Shape<Int<(Nthreads/nthreadsPerRow)>,Int<nthreadsPerRow>>,Stride<Int<nthreadsPerRow>,_1>>;
    auto smem_tiled_copy_scoresf16 = make_tiled_copy(SmemCopyAtomScoresf16{}, CopyLayoutScoresf16{}, Layout<Shape<_1,Int<SmemKAtomP/nthreadsPerRow>>>{});
    auto smem_thr_copy_scoresf16 = smem_tiled_copy_scoresf16.get_thread_slice(threadIdx.x);
    Tensor scores_f16_copypartition = smem_thr_copy_scoresf16.partition_D(sP_f16);


    using GmemCopyLayoutAtomV = Layout<Shape <Int<Nthreads / (SmemKAtomV/8)>, Int<SmemKAtomV/8>>,
                                  Stride<Int<SmemKAtomV/8>, _1>>;
    using GmemTiledCopyV = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
                        GmemCopyLayoutAtomV{},
                        Layout<Shape<_1, _8>>{})); 
    GmemTiledCopyV gmem_tiled_copy_V;
    auto gmem_thr_copy_V = gmem_tiled_copy_V.get_thread_slice(threadIdx.x);
    Tensor gV = make_tensor(make_gmem_ptr(Parameter_2_0_0+v_offset), Shape<Int<Bc>,Int<D>>{}, make_stride(Int<D>{},_1{}));
    // (32,256)
    Tensor gV1 = make_tensor(make_gmem_ptr(Parameter_2_0_0+v_offset), Shape<Int<BlockKSmem2>,Int<D>>{}, make_stride(Int<D>{},_1{}));
    Tensor gV_partition = gmem_thr_copy_V.partition_S(gV);
    Tensor gV1_partition = gmem_thr_copy_V.partition_S(gV1);
    // This has to be kBlockN and not 8, layout is: KSmem -> KblockN -> Kd(to be consistant with SmemLayoutV)
    using SmemLayoutAtomV = decltype(
        composition(Swizzle<kSwizzleV, 3, 3>{},
                    Layout<Shape<_8, Int<SmemKAtomV>>,
                           Stride<Int<SmemKAtomV>, _1>>{}));
    using SmemLayoutAtomVtransposedNoSwizzle = Layout<Shape<Int<SmemKAtomV>, Int<Bc>>,
                                                      Stride<_1, Int<SmemKAtomV>>>;
    using SmemLayoutAtomV1transposedNoSwizzle = Layout<Shape<Int<SmemKAtomV>, Int<BlockKSmem2>>,
                                                      Stride<_1, Int<SmemKAtomV>>>;
    using SmemLayoutAtomVtransposed = decltype(
        composition(Swizzle<kSwizzleV, 3, 3>{}, SmemLayoutAtomVtransposedNoSwizzle{}));
    using SmemLayoutAtomV1transposed = decltype(
        composition(Swizzle<kSwizzleV, 3, 3>{}, SmemLayoutAtomV1transposedNoSwizzle{}));
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<Bc>, Int<D>>{}));
    using SmemLayoutV1 = decltype(tile_to_shape(
        SmemLayoutAtomV{},
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
    
    Tensor sV_partition = gmem_thr_copy_V.partition_D(sV);
    Tensor sV1_partition = gmem_thr_copy_V.partition_D(sV1);
    Tensor rP = thr_mma.partition_fragment_A(sP_f16);
    Tensor rP1 = thr_mma.partition_fragment_A(sP_f161);
    auto smem_tiled_copy_P = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(threadIdx.x);
    Tensor sP_copypartition = smem_thr_copy_P.partition_S(sP_f16);
    Tensor sP1_copypartition = smem_thr_copy_P.partition_S(sP_f161);
    Tensor rP_copy_view = smem_thr_copy_P.retile_D(rP);
    Tensor rP1_copy_view = smem_thr_copy_P.retile_D(rP1);
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
    CopyAsyncV_g2s cp_g2s_v(gmem_tiled_copy_V, 
                      gV1_partition, sV1_partition, 
                      BlockKSmem2*D, size(sV1),num_stages_v);
    CopyAsyncV_g2s cp_g2s_mask(
                      gmem_tiled_copy_Mask, 
                      gMask_partition, sMask_partition, 
                      Bc, size(sMask),num_stages_mask);
    MatmulQK_s2r matmul_qk_s2r(smem_tiled_copy_Q, sQ1_copypartition, rQ1_copy_view, 
                               smem_tiled_copy_K, sK1_copypartition, rK1_copy_view, 
                               tiled_mma1, rQ1, rK1, acc_s_fragment, 
                               size(sQ1), size(sK1),num_stages_qk);
    // MatmulV_s2r matmul_v_s2r(smem_tiled_copy_V, sV1t_copypartition, rVt1_copy_view, 
    //                           tiled_mma, /*rP_Aregs,*/ rVt1, acc_o_fragment, 
    //                           size(sV1),num_stages_v);
    MatmulQsharedK_s2r matmul_v_s2r(smem_tiled_copy_P, sP1_copypartition, rP1_copy_view, 
                              smem_tiled_copy_V, sV1t_copypartition, rVt1_copy_view,
                              tiled_mma, rP1, rVt1, acc_o_fragment, 
                              size(sP_f161), size(sV1),num_stages_v);

    cp_g2s_qk.prologue();

    cute::fill(sr_old, 0.0f);
    cute::fill(r_new_fragment, 0.0f);
    cute::fill(r_wo_clamp_fragment, 0.0f);
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
    cp_g2s_mask.prologue();
    matmul_qk_s2r.epilogue();

    cp_async_wait_flash<0>();
    __syncthreads();
    cp_g2s_v.prologue();

    // qk*m
    multiply_mask(smem_tiled_copy_mask, sMask_copypartition, rMask_copy_view, acc_s_fragment, rMask);

    cute::copy(smem_tiled_copy_P1, rP_copy_view1, sP_copypartition1);

    __syncthreads();
    /*if(threadIdx.x == 0 & blockIdx.x==0){
      printf("\nsP\n");
      for(int i=0;i<size<0>(sP);i++){
        for(int j=0;j<size<1>(sP);j++){
          printf("%d,%d: %f ",i,j,sP(i,j));
        }
        printf("\n");
      }
    }*/
    
    cute::copy(smem_tiled_copy_P2, sP_copypartition2, scores_copy_view);

    // Tensor scores = make_tensor(acc_s_fragment.data(),convert_layout_scores(acc_s_fragment.layout()));// ((2,2)(Atom),M,N) -> ((2,M),(2,N))
    // r_new, r_wo_clamp
    update_r<8>(r_new_fragment, r_wo_clamp_fragment, scores);
    // qkm/r_new
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(scores); ax0++){
      #pragma unroll
      for(int ax1 = 0;ax1 < size<1>(scores); ax1++){
        scores(ax0,ax1) = scores(ax0,ax1) / r_new_fragment(ax0);
      }
    }

    #pragma unroll
    for(int ax0 = 0;ax0 < size(r_new_fragment);ax0++){
      // here causion!
      sr_new[threadIdx.x/nthreadsPerRow + ax0*(Nthreads/nthreadsPerRow)] = r_new_fragment(ax0);
    }

    cutlass::NumericArrayConverter<half, float, decltype(size(scores))::value> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float,decltype(size(scores))::value> *>(scores.data()));
    Tensor rP_f16 = make_tensor(make_rmem_ptr<half>(&frag), scores.layout());
    Tensor rP_f16_cp_view = make_tensor(rP_f16.data(),convert_layout_scores_copyview<SmemKAtomP/nthreadsPerRow>(rP_f16.layout()));
    Tensor rP_f16_copy_view = smem_thr_copy_scoresf16.retile_S(rP_f16_cp_view);
    cute::copy(smem_tiled_copy_scoresf16, rP_f16_copy_view, scores_f16_copypartition);
    
    __syncthreads();// r_new must be write out
    /*if(threadIdx.x==0 && blockIdx.x==0){
      printf("\nr_new\n");
      for(int i=0;i<size<0>(sr_new);i++){
        printf("%d: %f ",i,sr_new(i));
      }
      printf("\nscores\n");
      for(int i=0;i<size<0>(sP_f16);i++){
        for(int j=0;j<size<1>(scores);j++){
          printf("%f ",scores(i,j));
        }
        printf("\n");
      }
    }*/

    // Tensor rP_Aregs = make_tensor(rP.data(), convert_layout_rowcol_Aregs(rP.layout())); // ((2,M),(2,N)) -> ((2,2,2),M,N/2)
    // m_old, m_new, acc_o, p, v -> acc_o_new
    Tensor acc_o_rowcol = make_tensor(acc_o_fragment.data(),  convert_layout_scores(acc_o_fragment.layout()));


    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(acc_o_rowcol);ax0++){
      // here causion!
        float r_old_local = sr_old_copypartition(ax0%2*2,ax0/2,0);
        float r_new_local = sr_new_copypartition(ax0%2*2,ax0/2,0);
        float scale = r_old_local/r_new_local;
        #pragma unroll
        for(int ax1 = 0;ax1 < size<1>(acc_o_rowcol);ax1++){
            acc_o_rowcol(ax0,ax1) *= scale;
        }
    }

    #pragma unroll
    for(int ax0 = 0;ax0 < Bc/BlockKSmem2-1;ax0++){
       cp_async_wait_flash<0>();
      __syncthreads();
      cp_g2s_v.body();
      // matmul_v_s2r.body(rP_Aregs);
      matmul_v_s2r.body();
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
    // matmul_v_s2r.epilogue(rP_Aregs);
    matmul_v_s2r.epilogue();

    auto tmp = sr_new_copypartition.data();
    sr_new_copypartition.data() = sr_old_copypartition.data();
    sr_old_copypartition.data() = tmp;
    tmp = sr_new.data();
    sr_new.data() = sr_old.data();
    sr_old.data() = tmp;

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
    cp_g2s_mask.prologue();
    matmul_qk_s2r.epilogue();

    cp_async_wait_flash<0>();
    __syncthreads();
    cp_g2s_v.prologue();

    // qk*m
    multiply_mask(smem_tiled_copy_mask, sMask_copypartition, rMask_copy_view, acc_s_fragment, rMask);

    cute::copy(smem_tiled_copy_P1, rP_copy_view1, sP_copypartition1);

    __syncthreads();
    
    cute::copy(smem_tiled_copy_P2, sP_copypartition2, scores_copy_view);

    // Tensor scores = make_tensor(acc_s_fragment.data(),convert_layout_scores(acc_s_fragment.layout()));// ((2,2)(Atom),M,N) -> ((2,M),(2,N))
    // r_new, r_wo_clamp
    update_r<8>(r_new_fragment, r_wo_clamp_fragment, scores);
    // qkm/r_new
    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(scores); ax0++){
      #pragma unroll
      for(int ax1 = 0;ax1 < size<1>(scores); ax1++){
        scores(ax0,ax1) = scores(ax0,ax1) / r_new_fragment(ax0);
      }
    }

    #pragma unroll
    for(int ax0 = 0;ax0 < size(r_new_fragment);ax0++){
      // here causion!
      sr_new[threadIdx.x/nthreadsPerRow + ax0*(Nthreads/nthreadsPerRow)] = r_new_fragment(ax0);
    }

    cutlass::NumericArrayConverter<half, float, decltype(size(scores))::value> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float,decltype(size(scores))::value> *>(scores.data()));
    Tensor rP_f16 = make_tensor(make_rmem_ptr<half>(&frag), scores.layout());
    Tensor rP_f16_cp_view = make_tensor(rP_f16.data(),convert_layout_scores_copyview<SmemKAtomP/nthreadsPerRow>(rP_f16.layout()));
    Tensor rP_f16_copy_view = smem_thr_copy_scoresf16.retile_S(rP_f16_cp_view);
    cute::copy(smem_tiled_copy_scoresf16, rP_f16_copy_view, scores_f16_copypartition);
    
    __syncthreads();// r_new must be write out

    // Tensor rP_Aregs = make_tensor(rP.data(), convert_layout_rowcol_Aregs(rP.layout())); // ((2,M),(2,N)) -> ((2,2,2),M,N/2)
    // m_old, m_new, acc_o, p, v -> acc_o_new
    Tensor acc_o_rowcol = make_tensor(acc_o_fragment.data(),  convert_layout_scores(acc_o_fragment.layout()));


    #pragma unroll
    for(int ax0 = 0;ax0 < size<0>(acc_o_rowcol);ax0++){
      // here causion!
        float r_old_local = sr_old_copypartition(ax0%2*2,ax0/2,0);
        float r_new_local = sr_new_copypartition(ax0%2*2,ax0/2,0);
        float scale = r_old_local/r_new_local;
        #pragma unroll
        for(int ax1 = 0;ax1 < size<1>(acc_o_rowcol);ax1++){
            acc_o_rowcol(ax0,ax1) *= scale;
        }
    }

    #pragma unroll
    for(int ax0 = 0;ax0 < Bc/BlockKSmem2-1;ax0++){
       cp_async_wait_flash<0>();
      __syncthreads();
      cp_g2s_v.body();
      // matmul_v_s2r.body(rP_Aregs);
      matmul_v_s2r.body();
    }

     cp_async_wait_flash<0>();
    __syncthreads();
    // matmul_v_s2r.epilogue(rP_Aregs);
    matmul_v_s2r.epilogue();
    }
    // m_new,lse_new,acco -> out
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

    // r -> gR
    Tensor gR = make_tensor(make_gmem_ptr(r+lse_offset), Shape<Int<Br>>{}, make_stride(_1{}));
    if(threadIdx.x%nthreadsPerRow==0){
      #pragma unroll
      for(int ax0 = 0;ax0 < size<0>(r_new_fragment);ax0++){
        // here causion!
        gR(threadIdx.x/nthreadsPerRow + ax0*(Nthreads/nthreadsPerRow)) = r_new_fragment(ax0);
      }
    }

}
