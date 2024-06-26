// nvcc -std=c++17 -Xptxas=-v -lineinfo -O3 --use_fast_math -gencode=arch=compute_80,code=sm_80 -I ../csrc/ -I ../third_party/cutlass/include --disable-warnings -o attention attention.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include<thrust/device_vector.h>
#include<thrust/random.h>
#include<thrust/transform.h>
#include<thrust/iterator/counting_iterator.h>

#include <iostream>
#include <vector>

#include "configs.h"
#include "kernels/attention/smemfuse/smemfuse.h"
#include "kernels/attention/regfuse/regfuse.h"

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
float test_regfuse_attention(ProblemShape shape){
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

    int B = shape.B;
    int H = shape.H;
    int Seq_q = shape.Seq_q;
    int Seq_k = shape.Seq_k;

    int shared_mem = InplementConfig().shared_mem;

    auto kernel = &flashattn_fwd_regfuse<kd,d,br,bc,Nthreads,BlockKSmem,num_stages_qk,load_q_once,BlockKSmem2,num_stages_v,SmemKAtom,kSwizzle,unrollLastIter>;
    if(shared_mem > 48*1024){
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
    }

    thrust::device_vector<int> cache(int(256e6/4));
    //input argument
    thrust::device_vector<half> Parameter_0_0_0(B*H*Seq_q*kd);
    thrust::device_vector<half> Parameter_1_0_0(B*H*Seq_k*kd);
    thrust::device_vector<half> Parameter_2_0_0(B*H*Seq_k*d);
    //output argument
    thrust::device_vector<half> Result_7_0_0(B*H*Seq_q*d);
    thrust::counting_iterator<unsigned int>  index_begin(0);
    thrust::transform(index_begin, index_begin + B*H*Seq_q*kd, Parameter_0_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + B*H*Seq_k*kd, Parameter_1_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + B*H*Seq_k*d, Parameter_2_0_0.begin(), prg());
    
    auto q_ptr = thrust::raw_pointer_cast(Parameter_0_0_0.data());
    auto k_ptr = thrust::raw_pointer_cast(Parameter_1_0_0.data());
    auto v_ptr = thrust::raw_pointer_cast(Parameter_2_0_0.data());
    auto o_ptr = thrust::raw_pointer_cast(Result_7_0_0.data());

    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,o_ptr, H,Seq_k,Seq_q);
    cudaEventRecord(start, 0);
    for(int _ = 0; _ < 5; _++)
        kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,o_ptr, H,Seq_k,Seq_q);
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
        kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,o_ptr, H,Seq_k,Seq_q);
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
        kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,o_ptr, H,Seq_k,Seq_q);
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
float test_smemfuse_attention(ProblemShape shape){
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


    int B = shape.B;
    int H = shape.H;
    int Seq_q = shape.Seq_q;
    int Seq_k = shape.Seq_k;

    int shared_mem = InplementConfig().shared_mem;

    auto kernel = &flashattn_fwd_smemfuse<kd,d,br,bc,Nthreads,BlockKSmem,num_stages_qk,load_q_once,BlockKSmem2,num_stages_v,SmemKAtom,kSwizzle,SmemKAtomV,kSwizzleV,SmemKAtomP,kSwizzleP,SmemKAtomPf16,kSwizzlePf16,warps_mma1_N,warps_mma_N,unrollLastIter>;
    if(shared_mem > 48*1024){
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
    }

    thrust::device_vector<int> cache(int(256e6/4));
    //input argument
    thrust::device_vector<half> Parameter_0_0_0(B*H*Seq_q*kd);
    thrust::device_vector<half> Parameter_1_0_0(B*H*Seq_k*kd);
    thrust::device_vector<half> Parameter_2_0_0(B*H*Seq_k*d);
    //output argument
    thrust::device_vector<half> Result_7_0_0(B*H*Seq_q*d);
    thrust::counting_iterator<unsigned int>  index_begin(0);
    thrust::transform(index_begin, index_begin + B*H*Seq_q*kd, Parameter_0_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + B*H*Seq_k*kd, Parameter_1_0_0.begin(), prg());
    thrust::transform(index_begin, index_begin + B*H*Seq_k*d, Parameter_2_0_0.begin(), prg());
    
    auto q_ptr = thrust::raw_pointer_cast(Parameter_0_0_0.data());
    auto k_ptr = thrust::raw_pointer_cast(Parameter_1_0_0.data());
    auto v_ptr = thrust::raw_pointer_cast(Parameter_2_0_0.data());
    auto o_ptr = thrust::raw_pointer_cast(Result_7_0_0.data());

    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,o_ptr, H,Seq_k,Seq_q);
    cudaEventRecord(start, 0);
    for(int _ = 0; _ < 5; _++)
        kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,o_ptr, H,Seq_k,Seq_q);
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
        kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,o_ptr, H,Seq_k,Seq_q);
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
        kernel<<<dim3(B*H*Seq_q/br, 1, 1), dim3(Nthreads, 1, 1),shared_mem,0>>>(q_ptr,k_ptr,v_ptr,o_ptr, H,Seq_k,Seq_q);
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
    using InpleConfig = ImplementShapeAttnRegFwd<128,64,256,256,256>;
    float ms;
    // ms = test_regfuse_attention<InpleConfig>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;

    // ms = test_regfuse_attention<ImplementShapeAttnRegFwd<128,128,256,256,256,64,2,128,1,false>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    // ms = test_regfuse_attention<ImplementShapeAttnRegFwd<128,64,64,64,256>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    // ms = test_regfuse_attention<ImplementShapeAttnRegFwd<128,64,256,64,256>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    // ms = test_regfuse_attention<ImplementShapeAttnRegFwd<128,128,64,64,256>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    // ms = test_regfuse_attention<ImplementShapeAttnRegFwd<128,64,256,128,256>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;

    // A100-----------------------------------------
    // ms = test_smemfuse_attention<ImplementShapeAttnSharedFwd<64,64,256,256,128,2,4>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    // ms = test_smemfuse_attention<ImplementShapeAttnSharedFwd<128,32,256,256,256,2,4>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    // ms = test_smemfuse_attention<ImplementShapeAttnSharedFwd<64,64,256,384,256,2,8>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    // ms = test_smemfuse_attention<ImplementShapeAttnSharedFwd<128,64,256,384,256,2,4,64,2>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;  
    // ms = test_smemfuse_attention<ImplementShapeAttnSharedFwd<128,64,256,384,256,2,4,64,2,64,1,false>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;  
    // ms = test_smemfuse_attention<ImplementShapeAttnSharedFwd<128,64,256,128,256,2,2>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    // ms = test_smemfuse_attention<ImplementShapeAttnSharedFwd<64,64,256,512,256,2,8>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    // ms = test_smemfuse_attention<ImplementShapeAttnSharedFwd<64,64,256,128,256,2,4>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    // ms = test_smemfuse_attention<ImplementShapeAttnSharedFwd<64,64,256,128,256,2,4>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    // ms = test_smemfuse_attention<ImplementShapeAttnSharedFwd<32,64,256,128,128,2,4>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    // ms = test_smemfuse_attention<ImplementShapeAttnSharedFwd<64,64,256,128,128,2,2>>(PS);
    // std::cout << "Time: " << ms << "ms" << std::endl;
    
    // RTX 4090 -------------------------------------
    ms = test_smemfuse_attention<ImplementShapeAttnSharedFwd<64,32,256,128,128,1,2>>(PS);
    std::cout << "Time: " << ms << "ms" << std::endl;




    // int Batch = 4, Head = 8, Seqlen_q = 2048,seqlen_kv = 2048, dim_qk = 256, dim_v = 256;
    // constexpr float softmax_scale = 1.25e-1;
    // torch::TensorOptions option(torch::kFloat16);
    // torch::Tensor q = torch::randn({Batch, Head, Seqlen_q, dim_qk}, option).to(torch::kCUDA);
    // torch::Tensor k = torch::randn({Batch, Head, seqlen_kv, dim_qk}, option).to(torch::kCUDA);
    // torch::Tensor v = torch::randn({Batch, Head, seqlen_kv, dim_v}, option).to(torch::kCUDA);

    // torch::Tensor attn = torch::matmul(q, k.transpose(-2, -1)) / softmax_scale;
    // attn = attn.softmax(-1);
    // torch::Tensor out = torch::matmul(attn, v);
    // std::cout << out.sizes() << std::endl;



    return 0;
}
