// nvcc -std=c++17 -Xptxas=-v -lineinfo -O3 --use_fast_math -gencode=arch=compute_80,code=sm_80 -I ../csrc/ -I ../third_party/cutlass/include --disable-warnings -o attention attention.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include<thrust/device_vector.h>
#include<thrust/random.h>
#include<thrust/transform.h>
#include<thrust/iterator/counting_iterator.h>

#include <iostream>
#include <vector>

#include "../configs.h"
#include "kernels/retnet/smemfuse/smemfuse.h"
#include "kernels/retnet/regfuse/regfuse.h"
#include "kernels/retnet/recurrent/recurrent.h"

// #include <torch/all.h>


struct prg
{
    float a,b;
    __host__ __device__ 
    prg(float _a=0.f, float _b=1.f):a(_a),b(_b){};
    __host__ __device__ half operator()(const unsigned int n){
        thrust::default_random_engine rng;
        thrust::normal_distribution<float> dist(a,b);
        rng.discard(n);

        return half(dist(rng));
    }
};

template <typename InplementConfig>
float test_regfuse_retnet(ProblemShape shape){
    constexpr int br = InplementConfig::Br;
    constexpr int bc = InplementConfig::Bc;
    constexpr int kd = InplementConfig::Kd;
    constexpr int d = InplementConfig::D;
    constexpr int Nthreads = InplementConfig::Nthreads;
    constexpr int BlockKSmem = InplementConfig::BlockKSmem;
    constexpr int num_stages_qk = InplementConfig::num_stages_qk;
    constexpr bool load_q_once = InplementConfig::load_q_once;
    constexpr int BlockKSmem2 = InplementConfig::BlockKSmem2;
    constexpr int num_stages_v = InplementConfig::num_stages_v;
    constexpr int SmemKAtom = InplementConfig::SmemKAtom;
    constexpr int kSwizzle = InplementConfig::kSwizzle;
    constexpr bool unrollLastIter = InplementConfig::unrollLastIter;
    
    constexpr int num_stages_mask = InplementConfig::num_stages_mask;
    constexpr int SmemKAtomMask = InplementConfig::SmemKAtomMask;
    constexpr int kSwizzleMask = InplementConfig::kSwizzleMask;

    int B = shape.B;
    int H = shape.H;
    int Seq_q = shape.Seq_q;
    int Seq_k = shape.Seq_k;

    int shared_matmulqkv = num_stages_qk*(br)*BlockKSmem*sizeof(half)+num_stages_qk*bc*BlockKSmem*sizeof(half)+num_stages_v*BlockKSmem2* d* sizeof(half);
    int shared_mask = num_stages_mask*br*bc*sizeof(half);
    int shared_out = br * d * sizeof(half);
    int shared_mem = (shared_matmulqkv+shared_mask) > shared_out ? (shared_matmulqkv+shared_mask):shared_out;//(acc_o(p(q,k),v))

    // TODO: load q once
    auto kernel = &ret_fwd_regfuse<kd,d,br,bc,Nthreads,BlockKSmem,num_stages_qk,load_q_once,BlockKSmem2,num_stages_v, num_stages_mask, SmemKAtom,kSwizzle,SmemKAtomMask,kSwizzleMask,unrollLastIter>;
    if(shared_mem > 48*1024){
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
    }

    thrust::device_vector<int> cache(int(256e6/4));
    //input argument
    thrust::device_vector<half> Parameter_0_0_0(B*H*Seq_q*kd);
    thrust::device_vector<half> Parameter_1_0_0(B*H*Seq_k*kd);
    thrust::device_vector<half> Parameter_2_0_0(B*H*Seq_k*d);
    thrust::device_vector<half> Parameter_3_0_0(H*Seq_q*Seq_k);
    //output argument
    thrust::device_vector<half> Result_7_0_0(B*H*Seq_q*d);
    thrust::counting_iterator<unsigned int>  index_begin(0);
    thrust::transform(index_begin, index_begin + B*H*Seq_q*kd, Parameter_0_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + B*H*Seq_k*kd, Parameter_1_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + B*H*Seq_k*d, Parameter_2_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + H*Seq_q*Seq_k, Parameter_3_0_0.begin(), prg());
    
    auto q_ptr = thrust::raw_pointer_cast(Parameter_0_0_0.data());
    auto k_ptr = thrust::raw_pointer_cast(Parameter_1_0_0.data());
    auto v_ptr = thrust::raw_pointer_cast(Parameter_2_0_0.data());
    auto mask_ptr = thrust::raw_pointer_cast(Parameter_3_0_0.data());
    auto o_ptr = thrust::raw_pointer_cast(Result_7_0_0.data());

    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,mask_ptr,o_ptr, H,Seq_k,Seq_q);
    cudaEventRecord(start, 0);
    for(int _ = 0; _ < 5; _++)
        kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,mask_ptr,o_ptr, H,Seq_k,Seq_q);
    if(cudaEventRecord(stop, 0) != cudaSuccess) return -1;
    if(cudaEventSynchronize(stop) != cudaSuccess) return -1;
    if(cudaGetLastError() != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    cudaEventElapsedTime(&ms, start, stop);
    int warm_up = int(ceil(50.0 / (ms/5)));
    int repeats = int(ceil(100.0 / (ms/5)));
    for(int _ = 0; _ < warm_up; _++){
        kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,mask_ptr,o_ptr, H,Seq_k,Seq_q);
    }

    std::vector<cudaEvent_t> start_(repeats);
    std::vector<cudaEvent_t> stop_(repeats);
    for(int ii = 0; ii < repeats; ii++){
        cudaEventCreate(&start_[ii]);
        cudaEventCreate(&stop_[ii]);
    }
    for(int ii = 0; ii < repeats; ii++){
        thrust::fill(cache.begin(), cache.end(), ii);
        cudaEventRecord(start_[ii], 0);
        kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,mask_ptr,o_ptr, H,Seq_k,Seq_q);
        cudaEventRecord(stop_[ii], 0);
    }
    if(cudaEventSynchronize(stop_[repeats-1]) != cudaSuccess) return -1;
    if(cudaGetLastError() != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    ms = 0;
    for(int ii = 0; ii < repeats; ii++){
        float tmp;
        cudaEventElapsedTime(&tmp, start_[ii], stop_[ii]);
        ms += tmp;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop); 
    for(int ii = 0; ii < repeats; ii++){
        cudaEventDestroy(start_[ii]);
        cudaEventDestroy(stop_[ii]);
    }
    return ms / repeats;

}

template <typename InplementConfig>
float test_smemfuse_retnet(ProblemShape shape){
    constexpr int br = InplementConfig::Br;
    constexpr int bc = InplementConfig::Bc;
    constexpr int kd = InplementConfig::Kd;
    constexpr int d = InplementConfig::D;
    constexpr int Nthreads = InplementConfig::Nthreads;
    constexpr int BlockKSmem = InplementConfig::BlockKSmem;
    constexpr int num_stages_qk = InplementConfig::num_stages_qk;
    constexpr bool load_q_once = InplementConfig::load_q_once;
    constexpr int BlockKSmem2 = InplementConfig::BlockKSmem2;
    constexpr int num_stages_v = InplementConfig::num_stages_v;
    constexpr int SmemKAtom = InplementConfig::SmemKAtom;
    constexpr int kSwizzle = InplementConfig::kSwizzle;
    constexpr bool unrollLastIter = InplementConfig::unrollLastIter;

    constexpr int SmemKAtomV = InplementConfig::SmemKAtomV;
    constexpr int kSwizzleV = InplementConfig::kSwizzleV;
    constexpr int SmemKAtomP = InplementConfig::SmemKAtomP;
    constexpr int kSwizzleP = InplementConfig::kSwizzleP;
    constexpr int SmemKAtomPf16 = InplementConfig::SmemKAtomPf16;
    constexpr int kSwizzlePf16 = InplementConfig::kSwizzlePf16;
    constexpr int warps_mma1_N = InplementConfig::warps_mma1_N;
    constexpr int warps_mma_N = InplementConfig::warps_mma_N;

    constexpr int num_stages_mask = InplementConfig::num_stages_mask;
    constexpr int SmemKAtomMask = InplementConfig::SmemKAtomMask;
    constexpr int kSwizzleMask = InplementConfig::kSwizzleMask;

    int B = shape.B;
    int H = shape.H;
    int Seq_q = shape.Seq_q;
    int Seq_k = shape.Seq_k;

    int shared_matmulqkv = num_stages_qk*(br)*BlockKSmem*sizeof(half)+num_stages_qk*bc*BlockKSmem*sizeof(half)+num_stages_v*BlockKSmem2* d* sizeof(half);
    int shared_accs = br*bc*sizeof(float)+br*bc*sizeof(half) + 3*sizeof(float)*br;
    int shared_mask = num_stages_mask*br*bc*sizeof(half);
    int shared_out = br * d * sizeof(half);
    int shared_mem = (shared_matmulqkv+shared_accs+shared_mask) > shared_out ? (shared_matmulqkv+shared_accs+shared_mask):shared_out;//(acc_o(p(q,k),v))

    auto kernel = &ret_fwd_smemfuse<kd,d,br,bc,Nthreads,BlockKSmem,num_stages_qk/*,load_q_once*/,BlockKSmem2,num_stages_v,num_stages_mask,warps_mma1_N,warps_mma_N,SmemKAtom,kSwizzle,SmemKAtomV,kSwizzleV, SmemKAtomMask, kSwizzleMask,SmemKAtomP,kSwizzleP,SmemKAtomPf16,kSwizzlePf16,unrollLastIter>;
    if(shared_mem > 48*1024){
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
    }

    thrust::device_vector<int> cache(int(256e6/4));
    //input argument
    thrust::device_vector<half> Parameter_0_0_0(B*H*Seq_q*kd);
    thrust::device_vector<half> Parameter_1_0_0(B*H*Seq_k*kd);
    thrust::device_vector<half> Parameter_2_0_0(B*H*Seq_k*d);
    thrust::device_vector<half> Parameter_3_0_0(H*Seq_q*Seq_k);
    //output argument
    thrust::device_vector<half> Result_7_0_0(B*H*Seq_q*d);
    thrust::counting_iterator<unsigned int>  index_begin(0);
    thrust::transform(index_begin, index_begin + B*H*Seq_q*kd, Parameter_0_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + B*H*Seq_k*kd, Parameter_1_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + B*H*Seq_k*d, Parameter_2_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + H*Seq_q*Seq_k, Parameter_3_0_0.begin(), prg());

    auto q_ptr = thrust::raw_pointer_cast(Parameter_0_0_0.data());
    auto k_ptr = thrust::raw_pointer_cast(Parameter_1_0_0.data());
    auto v_ptr = thrust::raw_pointer_cast(Parameter_2_0_0.data());
    auto mask_ptr = thrust::raw_pointer_cast(Parameter_3_0_0.data());
    auto o_ptr = thrust::raw_pointer_cast(Result_7_0_0.data());

    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,mask_ptr,o_ptr, H,Seq_k,Seq_q);
    cudaEventRecord(start, 0);
    for(int _ = 0; _ < 5; _++)
        kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,mask_ptr,o_ptr, H,Seq_k,Seq_q);
    if(cudaEventRecord(stop, 0) != cudaSuccess) return -1;
    if(cudaEventSynchronize(stop) != cudaSuccess) return -1;
    if(cudaGetLastError() != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    cudaEventElapsedTime(&ms, start, stop);
    int warm_up = int(ceil(50.0 / (ms/5)));
    int repeats = int(ceil(100.0 / (ms/5)));
    for(int _ = 0; _ < warm_up; _++){
        kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,mask_ptr,o_ptr, H,Seq_k,Seq_q);
    }

    std::vector<cudaEvent_t> start_(repeats);
    std::vector<cudaEvent_t> stop_(repeats);
    for(int ii = 0; ii < repeats; ii++){
        cudaEventCreate(&start_[ii]);
        cudaEventCreate(&stop_[ii]);
    }
    for(int ii = 0; ii < repeats; ii++){
        thrust::fill(cache.begin(), cache.end(), ii);
        cudaEventRecord(start_[ii], 0);
        kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,mask_ptr,o_ptr, H,Seq_k,Seq_q);
        cudaEventRecord(stop_[ii], 0);
    }
    if(cudaEventSynchronize(stop_[repeats-1]) != cudaSuccess) return -1;
    if(cudaGetLastError() != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    ms = 0;
    for(int ii = 0; ii < repeats; ii++){
        float tmp;
        cudaEventElapsedTime(&tmp, start_[ii], stop_[ii]);
        ms += tmp;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop); 
    for(int ii = 0; ii < repeats; ii++){
        cudaEventDestroy(start_[ii]);
        cudaEventDestroy(stop_[ii]);
    }
    return ms / repeats;

}

float test_recurrent_retnet(recurrentShape shape){
    typedef float data_type;

    int B = shape.B;
    int H = shape.H;
    int Seq_q = shape.Seq_q;
    int Seq_k = shape.Seq_k;
    int dim_qk = shape.dim_qk;
    int dim_v = shape.dim_v;
    int block_dimqk = shape.block_dimqk;
    assert(Seq_q == Seq_k);
    assert(block_dimqk == dim_qk);

    thrust::device_vector<int> cache(int(256e6/4));
    thrust::device_vector<data_type> Parameter_0_0_0(B*H*Seq_q*dim_qk);
    thrust::device_vector<data_type> Parameter_1_0_0(B*H*Seq_k*dim_qk);
    thrust::device_vector<data_type> Parameter_2_0_0(B*H*Seq_k*dim_v);
    thrust::device_vector<data_type> Parameter_3_0_0(H*Seq_k);

    thrust::device_vector<data_type> Result_7_0_0(B*H*Seq_q*dim_v);
    thrust::counting_iterator<unsigned int>  index_begin(0);
    thrust::transform(index_begin, index_begin + B*H*Seq_q*dim_qk, Parameter_0_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + B*H*Seq_k*dim_qk, Parameter_1_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + B*H*Seq_k*dim_v, Parameter_2_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + H*Seq_k, Parameter_3_0_0.begin(), prg());

    auto q_ptr = thrust::raw_pointer_cast(Parameter_0_0_0.data());
    auto k_ptr = thrust::raw_pointer_cast(Parameter_1_0_0.data());
    auto v_ptr = thrust::raw_pointer_cast(Parameter_2_0_0.data());
    auto decay_ptr = thrust::raw_pointer_cast(Parameter_3_0_0.data());
    auto o_ptr = thrust::raw_pointer_cast(Result_7_0_0.data());

    auto funcc = &retnet_recurrent_fwd;

    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    funcc(q_ptr,k_ptr,v_ptr,decay_ptr,o_ptr,B,H,dim_qk,dim_v,Seq_k,block_dimqk);
    cudaEventRecord(start, 0);
    for(int _ = 0; _ < 5; _++)
        funcc(q_ptr,k_ptr,v_ptr,decay_ptr,o_ptr,B,H,dim_qk,dim_v,Seq_k,block_dimqk);
    if(cudaEventRecord(stop, 0) != cudaSuccess) return -1;
    if(cudaEventSynchronize(stop) != cudaSuccess) return -1;
    if(cudaGetLastError() != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    cudaEventElapsedTime(&ms, start, stop);
    int warm_up = int(ceil(50.0 / (ms/5)));
    int repeats = int(ceil(100.0 / (ms/5)));
    for(int _ = 0; _ < warm_up; _++){
        funcc(q_ptr,k_ptr,v_ptr,decay_ptr,o_ptr,B,H,dim_qk,dim_v,Seq_k,block_dimqk);
    }

    std::vector<cudaEvent_t> start_(repeats);
    std::vector<cudaEvent_t> stop_(repeats);
    for(int ii = 0; ii < repeats; ii++){
        cudaEventCreate(&start_[ii]);
        cudaEventCreate(&stop_[ii]);
    }
    for(int ii = 0; ii < repeats; ii++){
        thrust::fill(cache.begin(), cache.end(), ii);
        cudaEventRecord(start_[ii], 0);
        funcc(q_ptr,k_ptr,v_ptr,decay_ptr,o_ptr,B,H,dim_qk,dim_v,Seq_k,block_dimqk);
        cudaEventRecord(stop_[ii], 0);
    }
    if(cudaEventSynchronize(stop_[repeats-1]) != cudaSuccess) return -1;
    if(cudaGetLastError() != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    ms = 0;
    for(int ii = 0; ii < repeats; ii++){
        float tmp;
        cudaEventElapsedTime(&tmp, start_[ii], stop_[ii]);
        ms += tmp;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop); 
    for(int ii = 0; ii < repeats; ii++){
        cudaEventDestroy(start_[ii]);
        cudaEventDestroy(stop_[ii]);
    }
    return ms / repeats;

}
int main(){
    ProblemShape PS(4,8,2048,2048);
    using InpleConfig = ImplementShape<128,64,256,256,256>;
    float ms = test_regfuse_retnet<InpleConfig>(PS);
    std::cout << "Time: " << ms << "ms" << std::endl;

    // ms = test_regfuse_retnet<ImplementShape<128,128,256,256,256,1,1,64,2,128,1,64,false,64>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    // ms = test_regfuse_retnet<ImplementShape<128,64,64,64,256>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    // ms = test_regfuse_retnet<ImplementShape<128,64,256,64,256>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    // ms = test_regfuse_retnet<ImplementShape<128,128,64,64,256>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    ms = test_regfuse_retnet<ImplementShape<128,64,256,128,256>>(PS);
    std::cout << "Time: " << ms << "ms" << std::endl;


    ms = test_smemfuse_retnet<ImplementShape<64,64,256,256,256,2,4>>(PS);
    std::cout << "Time: " << ms << "ms" << std::endl;

    recurrentShape PS2(8,32,2048,2048,128,128);
    ms = test_recurrent_retnet(PS2);
    std::cout << "Time: " << ms << "ms" << std::endl;

    return 0;
}
