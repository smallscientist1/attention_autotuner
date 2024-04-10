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
#include "kernels/retnet/bwd/bwd_colblock.h"

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
float test_retnet_bwd(ProblemShape shape){
    constexpr int Br = InplementConfig::Br;
    constexpr int Bc = InplementConfig::Bc;
    constexpr int Kd = InplementConfig::Kd;
    constexpr int D = InplementConfig::D;
    constexpr int Nthreads = InplementConfig::Nthreads;
    constexpr int BlockKSmem = InplementConfig::BlockKSmem;
    constexpr int num_stages_qk = InplementConfig::num_stages_qk;
    constexpr bool load_q_once = InplementConfig::load_q_once;
    constexpr int SmemKAtom = InplementConfig::SmemKAtom;
    constexpr int kSwizzle = InplementConfig::kSwizzle;
    constexpr bool unrollLastIter = InplementConfig::unrollLastIter;

    constexpr int mmawarpsN = InplementConfig::mmawarpsN;
    constexpr int mmawarpsN_dv = InplementConfig::mmawarpsN_dv;
    constexpr int mmawarpsN_dk = InplementConfig::mmawarpsN_dk;
    constexpr int mmawarpsN_dq = InplementConfig::mmawarpsN_dq;
    constexpr int num_stages_dv = InplementConfig::num_stages_dv;
    constexpr int num_stages_ds = InplementConfig::num_stages_ds;
    constexpr int num_stages_dq = InplementConfig::num_stages_dq;
    constexpr int num_stages_mask = InplementConfig::num_stages_mask;

    constexpr int SmemKAtomV = InplementConfig::SmemKAtomV;
    constexpr int kSwizzleV = InplementConfig::kSwizzleV;
    constexpr int SmemKAtomMask = InplementConfig::SmemKAtomMask;
    constexpr int kSwizzleMask = InplementConfig::kSwizzleMask;
    constexpr int SmemKAtomO = InplementConfig::SmemKAtomO;
    constexpr int kSwizzleO = InplementConfig::kSwizzleO;
    constexpr int SmemKAtomS = InplementConfig::SmemKAtomS;
    constexpr int kSwizzleS = InplementConfig::kSwizzleS;

    

    int B = shape.B;
    int H = shape.H;
    int Seq_q = shape.Seq_q;
    int Seq_k = shape.Seq_k;

    int shared_matmulqk = num_stages_qk*(Br)*BlockKSmem*sizeof(half)+num_stages_qk*Bc*BlockKSmem*sizeof(half);
    int shared_mask = num_stages_mask*Br*Bc*sizeof(half);
    int shared_SdO = Br*Bc*sizeof(half)+Br*D*sizeof(half);
    int shared_v = Bc*D*sizeof(half);
    int shared_mem = shared_matmulqk+shared_mask+shared_SdO+shared_v;
    
    int shared_mem_convert_dq = Br*Kd*sizeof(half);

    auto kernel1 = &ret_bwd_colblock<Kd,D,Br,Bc,Nthreads,mmawarpsN,mmawarpsN_dv,mmawarpsN_dk,mmawarpsN_dq,BlockKSmem, num_stages_qk, load_q_once, num_stages_mask, num_stages_dv, num_stages_ds, num_stages_dq, SmemKAtom, kSwizzle, SmemKAtomS, kSwizzleS, SmemKAtomO, kSwizzleO, SmemKAtomMask, kSwizzleMask,SmemKAtomV, kSwizzleV, unrollLastIter>;
    auto kernel2 = &convert_dq<Kd,Br,Bc,Nthreads,mmawarpsN_dq,SmemKAtom,kSwizzle>;
    if(shared_mem > 48*1024){
        cudaFuncSetAttribute(kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
    }
    if(shared_mem_convert_dq > 48*1024){
        cudaFuncSetAttribute(kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_convert_dq);
    }

    thrust::device_vector<int> cache(int(256e6/4));
    //input argument
    thrust::device_vector<half> Parameter_0_0_0(B*H*Seq_q*Kd);
    thrust::device_vector<half> Parameter_1_0_0(B*H*Seq_k*Kd);
    thrust::device_vector<half> Parameter_2_0_0(B*H*Seq_k*D);
    thrust::device_vector<half> Parameter_3_0_0(H*Seq_q*Seq_k);
    thrust::device_vector<float> r(B*H*Seq_q);
    thrust::device_vector<half> Result_7_0_0(B*H*Seq_q*D);

    //output argument
    thrust::device_vector<half> dq(B*H*Seq_q*Kd);
    thrust::device_vector<half> dk(B*H*Seq_k*Kd);
    thrust::device_vector<half> dv(B*H*Seq_k*D);

    thrust::counting_iterator<unsigned int>  index_begin(0);
    thrust::transform(index_begin, index_begin + B*H*Seq_q*Kd, Parameter_0_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + B*H*Seq_k*Kd, Parameter_1_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + B*H*Seq_k*D, Parameter_2_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + H*Seq_q*Seq_k, Parameter_3_0_0.begin(), prg());
    // TODO: compute r,result
    thrust::transform(index_begin, index_begin + B*H*Seq_q, r.begin(), prg());
    thrust::transform(index_begin, index_begin + B*H*Seq_q*D, Result_7_0_0.begin(), prg());


    auto q_ptr = thrust::raw_pointer_cast(Parameter_0_0_0.data());
    auto k_ptr = thrust::raw_pointer_cast(Parameter_1_0_0.data());
    auto v_ptr = thrust::raw_pointer_cast(Parameter_2_0_0.data());
    auto mask_ptr = thrust::raw_pointer_cast(Parameter_3_0_0.data());
    auto r_ptr = thrust::raw_pointer_cast(r.data());
    auto o_ptr = thrust::raw_pointer_cast(Result_7_0_0.data());
    auto dq_ptr = thrust::raw_pointer_cast(dq.data());
    auto dk_ptr = thrust::raw_pointer_cast(dk.data());
    auto dv_ptr = thrust::raw_pointer_cast(dv.data());


    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* dqaccum;
    cudaMalloc(&dqaccum, B*H*Seq_q*Kd*sizeof(float));
    cudaMemset(dqaccum, 0, B*H*Seq_q*Kd*sizeof(float));
    kernel1<<<dim3(B*H*Seq_k/Bc, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,mask_ptr,o_ptr,r_ptr,dq_ptr,dk_ptr,dv_ptr,dqaccum, H,Seq_k,Seq_q);
    kernel2<<<dim3(B*H*Seq_q/Br, 1, 1), dim3(Nthreads, 1, 1),shared_mem_convert_dq,0>>>(dq_ptr,dqaccum,Seq_k,Seq_q);
    cudaFree(dqaccum);
    cudaEventRecord(start, 0);
    return 0;
    /*
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
    return ms / repeats;*/

}


int main(){
    ProblemShape PS(4,8,2048,2048);
    using InpleConfig = ImplementShapeBwd<64,64,256,256,256,2,4,4,4>;
    float ms = test_retnet_bwd<InpleConfig>(PS);
    std::cout << "Time: " << ms << "ms" << std::endl;


    return 0;
}
