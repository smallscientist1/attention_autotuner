global_vars = \
"""
#include "smemfuse.h"

constexpr int Br = {Br};
constexpr int Bc = {Bc};
constexpr int Kd = {Kd};
constexpr int D = {D};

constexpr bool unrollLastIter = {unrollLastIter};
// for q&k splitk
constexpr int BlockKSmem = {BlockKSmem};
constexpr int num_stages_qk = {num_stages_qk};
constexpr bool load_q_once = (BlockKSmem == Kd);
constexpr int num_stages_mask = {num_stages_mask};
// for V splitk
constexpr int BlockKSmem2 = {BlockKSmem2};
constexpr int num_stages_v = {num_stages_v};
// Nthreads
constexpr int Nthreads = {Nthreads};
// mma1 N
constexpr int warps_mma1_N = {warps_mma1_n};
// mma N
constexpr int warps_mma_N = {warps_mma_n};
// for sQ,sK swizzle
constexpr int SmemKAtom = BlockKSmem % 64 == 0 ? 64 : 32;
constexpr int kSwizzle = SmemKAtom == 32 ? 2 : 3;
// for sV swizzle
constexpr int SmemKAtomV = 64;
constexpr int kSwizzleV = SmemKAtomV == 32 ? 2 : 3;
// for sP swizzle
constexpr int SmemKAtomP = Bc % 64 == 0 ? 64 : 32;
constexpr int kSwizzleP = SmemKAtomP == 32 ? 2 : 3;
// for sP_f16 swizzle
constexpr int SmemKAtomPf16 = BlockKSmem2 % 64 == 0 ? 64 : 32;
constexpr int kSwizzlePf16 = SmemKAtomPf16 == 32 ? 2 : 3;
// for mask swizzlw
constexpr int SmemKAtomMask = Bc % 64 == 0 ? 64 : 32;
constexpr int kSwizzleMask = SmemKAtomMask == 32 ? 2 : 3;

constexpr int shared_matmulqkv = num_stages_qk*(Br)*BlockKSmem*sizeof(half)+num_stages_qk*Bc*BlockKSmem*sizeof(half)+num_stages_v*BlockKSmem2*D*sizeof(half);
constexpr int shared_accs = Br*Bc*sizeof(float)+Br*Bc*sizeof(half) + 2*sizeof(float)*Br;
constexpr int shared_mask = num_stages_mask*Br*Bc*sizeof(half);
constexpr int shared_out = Br * D * sizeof(half);
constexpr int shared_mem = (shared_matmulqkv + shared_accs + shared_mask) > shared_out ? (shared_matmulqkv + shared_accs + shared_mask):shared_out;//(acc_o(p(q,k),v))

"""

profile_func = \
"""
extern "C" float profile(half* Parameter_0_0_0,half* Parameter_1_0_0,half* Parameter_2_0_0, half* Parameter_3_0_0,half* Result_7_0_0,int B,int H, int Seq_q, int Seq_k){{
    auto kernel = &ret_fwd_smemfuse<Kd,D,Br,Bc,Nthreads,BlockKSmem,num_stages_qk,load_q_once,BlockKSmem2,num_stages_v,num_stages_mask,warps_mma1_N,warps_mma_N,SmemKAtom,kSwizzle, SmemKAtomV,kSwizzleV ,SmemKAtomMask,kSwizzleMask, SmemKAtomP, kSwizzleP, SmemKAtomPf16, kSwizzlePf16,unrollLastIter>;
    if(shared_mem > 48*1024){{
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
    }}

    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    kernel<<<dim3(B*H*Seq_q/Br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(Parameter_0_0_0,Parameter_1_0_0,Parameter_2_0_0,Parameter_3_0_0,Result_7_0_0, H,Seq_k,Seq_q);
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
        kernel<<<dim3(B*H*Seq_q/Br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(Parameter_0_0_0,Parameter_1_0_0,Parameter_2_0_0,Parameter_3_0_0,Result_7_0_0, H,Seq_k,Seq_q);
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
extern "C" int kernel_entry(half* Parameter_0_0_0, half* Parameter_1_0_0, half* Parameter_2_0_0, half* Parameter_3_0_0, half* Result_7_0_0, int B, int H, int Seq_k,int Seq_q)
{{
    auto kernel = &ret_fwd_smemfuse<Kd,D,Br,Bc,Nthreads,BlockKSmem,num_stages_qk,load_q_once,BlockKSmem2,num_stages_v,num_stages_mask,warps_mma1_N,warps_mma_N,SmemKAtom,kSwizzle, SmemKAtomV,kSwizzleV ,SmemKAtomMask,kSwizzleMask, SmemKAtomP, kSwizzleP, SmemKAtomPf16, kSwizzlePf16,unrollLastIter>;
  if(shared_mem > 48*1024){{
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
  }}
    kernel<<<dim3(B*H*Seq_q/Br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(Parameter_0_0_0,Parameter_1_0_0,Parameter_2_0_0,Parameter_3_0_0,Result_7_0_0, H,Seq_k,Seq_q);
    return 0;
}}
"""
profile_code =  global_vars + profile_func
kernel_code = global_vars + kernel_entry
