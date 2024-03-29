global_vars = \
"""
#include "regfuse.h"

constexpr int Br = {Br};
constexpr int Bc = {Bc};
constexpr int Kd = {Kd};
constexpr int D = {D};

constexpr bool unrollLastIter = {unrollLastIter};
// for q&k splitk
__device__ constexpr int BlockKSmem = {BlockKSmem};
constexpr int num_stages_qk = {num_stages_qk};
constexpr bool load_q_once = (BlockKSmem == Kd);
// for V splitk
constexpr int BlockKSmem2 = {BlockKSmem2};
constexpr int num_stages_v = {num_stages_v};
// for sQ,sK,sV,sO swizzle
constexpr int SmemKAtom = BlockKSmem % 64 == 0 ? 64 : 32;
constexpr int kSwizzle = SmemKAtom == 32 ? 2 : 3;

constexpr int shared_matmulqkv = num_stages_qk*(Br)*BlockKSmem*sizeof(half)+num_stages_qk*Bc*BlockKSmem*sizeof(half)+num_stages_v*BlockKSmem2*D*sizeof(half);
constexpr int shared_out = Br * D * sizeof(half);
constexpr int shared_mem = (shared_matmulqkv) > shared_out ? (shared_matmulqkv):shared_out;//(acc_o(p(q,k),v))

constexpr int Nthreads = {Nthreads};
"""
profile_func = \
"""
extern "C" float profile(half* Parameter_0_0_0,half* Parameter_1_0_0,half* Parameter_2_0_0, half* Result_7_0_0,int B,int H, int Seq_q, int Seq_k){{
    auto kernel = &flashattn_fwd_regfuse<Kd,D,Br,Bc,Nthreads,BlockKSmem,num_stages_qk,load_q_once,BlockKSmem2,num_stages_v,SmemKAtom,kSwizzle,unrollLastIter>;
    if(shared_mem > 48*1024){{
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
    }}

    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    kernel<<<dim3(B*H*Seq_q/Br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(Parameter_0_0_0,Parameter_1_0_0,Parameter_2_0_0,Result_7_0_0, H,Seq_k,Seq_q);
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
        kernel<<<dim3(B*H*Seq_q/Br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(Parameter_0_0_0,Parameter_1_0_0,Parameter_2_0_0,Result_7_0_0, H,Seq_k,Seq_q);
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
extern "C" int kernel_entry(half* Parameter_0_0_0, half* Parameter_1_0_0, half* Parameter_2_0_0, half* Result_7_0_0, int B, int H, int Seq_k,int Seq_q)
{{
    auto kernel = &flashattn_fwd_regfuse<Kd,D,Br,Bc,Nthreads,BlockKSmem,num_stages_qk,load_q_once,BlockKSmem2,num_stages_v,SmemKAtom,kSwizzle,unrollLastIter>;
  if(shared_mem > 48*1024){{
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
  }}
    kernel<<<dim3(B*H*Seq_q/Br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(Parameter_0_0_0,Parameter_1_0_0,Parameter_2_0_0,Result_7_0_0, H,Seq_k,Seq_q);
    return 0;
}}
"""
profile_code = global_vars + profile_func
kernel_code = global_vars + kernel_entry
