global_vars = \
"""
#include "bwd_colblock.h"

    constexpr int Br = {Br};
    constexpr int Bc = {Bc};
    constexpr int Kd = {Kd};
    constexpr int D = {D};
    constexpr bool unrollLastIter = {unrollLastIter};
    constexpr int Nthreads = {Nthreads};
    // for q&k splitk
    constexpr int BlockKSmem = {BlockKSmem};
    constexpr int num_stages_qk = {num_stages_qk};
    constexpr bool load_q_once = (BlockKSmem == Kd);
    // num_stages
    constexpr int num_stages_mask = {num_stages_mask};
    constexpr int num_stages_dv = {num_stages_dv};
    constexpr int num_stages_ds = {num_stages_ds};
    constexpr int num_stages_dq = {num_stages_dq};
    // for sQ,sK swizzle
    constexpr int SmemKAtom = BlockKSmem % 64 == 0 ? 64 : 32;
    constexpr int kSwizzle = SmemKAtom == 32 ? 2 : 3;
    // for sMask swizzle
    constexpr int SmemKAtomMask = Bc % 64 == 0 ? 64 : 32;
    constexpr int kSwizzleMask = SmemKAtomMask == 32 ? 2 : 3;
    // for sS swizzle
    constexpr int SmemKAtomS = BlockKSmem % 64 == 0 ? 64 : 32;
    constexpr int kSwizzleS = SmemKAtomS == 32 ? 2 : 3;
    // for sO swizzle
    constexpr int SmemKAtomO = BlockKSmem % 64 == 0 ? 64 : 32;
    constexpr int kSwizzleO = SmemKAtomO == 32 ? 2 : 3;
    // for sV swizzle
    constexpr int SmemKAtomV = D % 64 == 0 ? 64 : 32;
    constexpr int kSwizzleV = SmemKAtomV == 32 ? 2 : 3;
    // mmawarpsN
    constexpr int mmawarpsN = {mmawarpsN};
    constexpr int mmawarpsN_dv = {mmawarpsN_dv};
    constexpr int mmawarpsN_dk = {mmawarpsN_dk};
    constexpr int mmawarpsN_dq = {mmawarpsN_dq};

    constexpr int shared_matmulqk = num_stages_qk*(Br)*BlockKSmem*sizeof(half)+num_stages_qk*Bc*BlockKSmem*sizeof(half);
    constexpr int shared_mask = num_stages_mask*Br*Bc*sizeof(half);
    constexpr int shared_SdO = Br*Bc*sizeof(half)+Br*D*sizeof(half);
    constexpr int shared_v = Bc*D*sizeof(half);
    constexpr int shared_mem = shared_matmulqk+shared_mask+shared_SdO+shared_v;

    constexpr int shared_mem_convert_dq = Br*Kd*sizeof(half);

"""

profile_func = \
"""
extern "C" float profile(half* Parameter_0_0_0,half* Parameter_1_0_0,half* Parameter_2_0_0, half* Parameter_3_0_0,half* Result_7_0_0, float* r,half* dq, half* dk, half* dv, float* dqaccum, int B,int H, int Seq_q, int Seq_k){{
    auto kernel1 = &ret_bwd_colblock<Kd,D,Br,Bc,Nthreads,mmawarpsN,mmawarpsN_dv,mmawarpsN_dk,mmawarpsN_dq,BlockKSmem, num_stages_qk, load_q_once, num_stages_mask, num_stages_dv, num_stages_ds, num_stages_dq, SmemKAtom, kSwizzle, SmemKAtomS, kSwizzleS, SmemKAtomO, kSwizzleO, SmemKAtomMask, kSwizzleMask,SmemKAtomV, kSwizzleV, unrollLastIter>;
    auto kernel2 = &convert_dq<Kd,Br,Bc,Nthreads,mmawarpsN_dq,SmemKAtom,kSwizzle>;
    if(shared_mem > 48*1024){{
      cudaFuncSetAttribute(kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
    }}
    if(shared_mem_convert_dq > 48*1024){{
      cudaFuncSetAttribute(kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_convert_dq);
    }}

    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    kernel1<<<dim3(B*H*Seq_k/Bc, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(Parameter_0_0_0,Parameter_1_0_0,Parameter_2_0_0,Parameter_3_0_0,Result_7_0_0,r,dq,dk,dv,dqaccum, H,Seq_k,Seq_q);
    kernel2<<<dim3(B*H*Seq_q/Br, 1, 1), dim3(Nthreads, 1, 1),shared_mem_convert_dq,0>>>(dq,dqaccum,Seq_k,Seq_q);
    if(cudaEventRecord(stop, 0) != cudaSuccess) return -1;
    if(cudaEventSynchronize(stop) != cudaSuccess) return -1;
    if(cudaGetLastError() != cudaSuccess) {{
        printf("CUDA error: %s\\n", cudaGetErrorString(cudaGetLastError()));
        return -1;
    }}
    cudaEventElapsedTime(&ms, start, stop);
    int repeats = int(ceil(100.0 / ms));
    cudaEventRecord(start, 0);
    for(int _ = 0; _ < repeats; _++){{
    kernel1<<<dim3(B*H*Seq_k/Bc, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(Parameter_0_0_0,Parameter_1_0_0,Parameter_2_0_0,Parameter_3_0_0,Result_7_0_0,r,dq,dk,dv,dqaccum, H,Seq_k,Seq_q);
    kernel2<<<dim3(B*H*Seq_q/Br, 1, 1), dim3(Nthreads, 1, 1),shared_mem_convert_dq,0>>>(dq,dqaccum,Seq_k,Seq_q);
    }}
    if(cudaEventRecord(stop, 0) != cudaSuccess) return -1;
    if(cudaEventSynchronize(stop) != cudaSuccess) return -1;
    if(cudaGetLastError() != cudaSuccess) {{
        printf("CUDA error: %s\\n", cudaGetErrorString(cudaGetLastError()));
        return -1;
    }}
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop); 
    return ms / repeats;
}}
"""

kernel_entry = \
"""
extern "C" int kernel_entry(half* Parameter_0_0_0, half* Parameter_1_0_0, half* Parameter_2_0_0, half* Parameter_3_0_0, half* Result_7_0_0,float* r, half* dq, half* dk, half* dv, float* dqaccum, int B, int H, int Seq_k,int Seq_q)
{{
    auto kernel1 = &ret_bwd_colblock<Kd,D,Br,Bc,Nthreads,mmawarpsN,mmawarpsN_dv,mmawarpsN_dk,mmawarpsN_dq,BlockKSmem, num_stages_qk, load_q_once, num_stages_mask, num_stages_dv, num_stages_ds, num_stages_dq, SmemKAtom, kSwizzle, SmemKAtomS, kSwizzleS, SmemKAtomO, kSwizzleO, SmemKAtomMask, kSwizzleMask,SmemKAtomV, kSwizzleV, unrollLastIter>;
    auto kernel2 = &convert_dq<Kd,Br,Bc,Nthreads,mmawarpsN_dq,SmemKAtom,kSwizzle>;
  if(shared_mem > 48*1024){{
    cudaFuncSetAttribute(kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
  }}
  if(shared_mem_convert_dq > 48*1024){{
    cudaFuncSetAttribute(kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_convert_dq);
  }}
  // float* dqaccum;
  // CUDA_SAFE_CALL(cudaMalloc((void**)&dqaccum, sizeof(float)*B*H*Seq_q*Kd));
  // CUDA_SAFE_CALL(cudaMemset(dqaccum, 0, sizeof(float)*B*H*Seq_q*Kd));
  kernel1<<<dim3(B*H*Seq_k/Bc, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(Parameter_0_0_0,Parameter_1_0_0,Parameter_2_0_0,Parameter_3_0_0,Result_7_0_0,r,dq,dk,dv,dqaccum, H,Seq_k,Seq_q);
  kernel2<<<dim3(B*H*Seq_q/Br, 1, 1), dim3(Nthreads, 1, 1),shared_mem_convert_dq,0>>>(dq,dqaccum,Seq_k,Seq_q);
  // CUDA_SAFE_CALL(cudaFree(dqaccum));
    return 0;
}}
"""

profile_code = global_vars + profile_func
kernel_code = global_vars + kernel_entry
