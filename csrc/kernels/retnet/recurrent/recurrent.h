// nvcc -std=c++17 -lineinfo -O3 -Xptxas=-v --use_fast_math --expt-relaxed-constexpr --disable-warnings --compiler-options '-fPIC' --shared scan_cpasync.cu -lcuda -gencode=arch=compute_80,code=sm_80 -o scan_cpasync.so -I /home/v-feiychen/flashfusion/cutlass_cute/include
#define CUB_STDERR

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>

using namespace cub;
using scan_t = float2;
using data_t = float;
#define MAX_KDIM 256
#define MAX_VDIM 16

template<int BYTES> struct BytesToType {};

template<> struct BytesToType<16> {
    using Type = uint4;
    static_assert(sizeof(Type) == 16);
};

template<> struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
};

template<> struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};

template<> struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};

template<> struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1);
};

struct Params {
void *__restrict__ d_k;         // Tile of input
void *__restrict__ d_v;         // Tile of input
void *__restrict__ d_decay;       // Tile of input
void *__restrict__ d_out;         // Tile of output
void *__restrict__ d_q;  
int batch, heads, vdim, kdim, l;
int k_bh_stride, k_kdim_stride;
int v_bh_stride, v_vdim_stride;
int q_bh_stride, q_kdim_stride;
int out_bh_stride, out_vdim_stride;
int decay_head_stride;
int block_kdim, block_vdim;

int nchunk, chunk_size;
};
//---------------------------------------------------------------------
// Kernel traits
//---------------------------------------------------------------------
template <
    int                     BLOCK_THREADS_,
    int                     ITEMS_PER_THREAD_,
    BlockScanAlgorithm      ALGORITHM_,
    int                     kNvdim_,
    typename data_t_, typename scan_t_>
struct scan_fwd_kernel_traits
{
    using data_t = data_t_;
    using scan_t = scan_t_;
    static constexpr int kNBytes = sizeof(data_t);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;//std::min(8, kNItems);
    static constexpr int kNLoads = ITEMS_PER_THREAD_ / kNElts;
    using vec_t = typename BytesToType<sizeof(data_t) * kNElts>::Type;
    static constexpr int BLOCK_THREADS = BLOCK_THREADS_;
    static constexpr int ITEMS_PER_THREAD = ITEMS_PER_THREAD_;
    static constexpr BlockScanAlgorithm ALGORITHM = ALGORITHM_;
    // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
    using BlockLoadT = BlockLoad<data_t, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadQKT = BlockLoad<data_t, BLOCK_THREADS, ITEMS_PER_THREAD * 16, BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = BlockLoad<vec_t, BLOCK_THREADS, kNLoads, BLOCK_LOAD_WARP_TRANSPOSE>;


    // Specialize BlockStore type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
    using BlockStoreT = BlockStore<data_t, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_WARP_TRANSPOSE>;
    
    // Specialize BlockScan type for our thread block
    using BlockScanT = BlockScan<scan_t, BLOCK_THREADS, ALGORITHM>;

    static constexpr int kChunkSize = BLOCK_THREADS * ITEMS_PER_THREAD;
    static constexpr int kSmemLoadqkSize = 2* sizeof(data_t)*kChunkSize;
    static constexpr int kSmemLoadSize = std::max(sizeof(typename BlockLoadT::TempStorage), (unsigned long)(kSmemLoadqkSize));
    static constexpr int kSmemStoreSize = sizeof(typename BlockStoreT::TempStorage);
    static constexpr int kSmemIOSize = std::max(kSmemLoadSize, kSmemStoreSize);
    static constexpr int kSmemScanSize = sizeof(typename BlockScanT::TempStorage);
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);

    static constexpr int kNvdim = kNvdim_;

};

//---------------------------------------------------------------------
// Kernels
//---------------------------------------------------------------------
struct ScanOp {
    __device__ __forceinline__ scan_t operator()(const scan_t &a, const scan_t &b) const {
        return make_float2(a.x * b.x, a.y * b.x + b.y);
    }
};
struct BlockPrefixCallbackOp
{
    // Running prefix
    scan_t running_prefix;
    // Constructor
    __device__ BlockPrefixCallbackOp(scan_t running_prefix) : running_prefix(running_prefix) {}
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ scan_t operator()(scan_t block_aggregate)
    {
        
        scan_t old_prefix = running_prefix;
        running_prefix = ScanOp()(old_prefix, block_aggregate);
        return old_prefix;
    }
};

template <typename Ktrait>
__global__ void scan_fwd_kernel( Params params
    // data_t         *d_in,          // Tile of input
    // data_t          *d_decay,       // Tile of input
    // data_t         *d_out,         // Tile of output
    // // clock_t     *d_elapsed,  // Elapsed cycle count of block scan
    // int batch, int heads, int vdim, int kdim, int l,
    // int kv_bh_stride, int kv_kdim_stride, int kv_vdim_stride,
    // int out_bh_stride, int out_kdim_stride, int out_vdim_stride,
    // int decay_head_stride,
    // int block_kdim, int block_vdim
    )    
{

    using data_t = typename Ktrait::data_t;
    using scan_t = typename Ktrait::scan_t;
    using vec_t = typename Ktrait::vec_t;
    constexpr int ITEMS_PER_THREAD = Ktrait::ITEMS_PER_THREAD;
    constexpr int NUM_THREADS = Ktrait::BLOCK_THREADS;
    constexpr int chunk_size = Ktrait::kChunkSize;
    constexpr int kNvdim = Ktrait::kNvdim;

    extern __shared__ char smem[];
    auto& smem_load = reinterpret_cast<typename Ktrait::BlockLoadT::TempStorage&>(smem);
    auto& smem_store = reinterpret_cast<typename Ktrait::BlockStoreT::TempStorage&>(smem);
    auto& smem_scan = *reinterpret_cast<typename Ktrait::BlockScanT::TempStorage*>(smem + Ktrait::kSmemIOSize);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t*>(smem + Ktrait::kSmemSize);

    char* smem_q = smem + 0;
    char* smem_k = smem + chunk_size * sizeof(data_t);

    // const int bh_id = blockIdx.x;
    // const int vdim_id = blockIdx.y;
    const int bh_id = blockIdx.y;
    const int vdim_id = blockIdx.x;
    const int h_id = bh_id % params.heads;
    
    data_t *decay = reinterpret_cast<data_t *>(params.d_decay) + h_id * params.decay_head_stride; 
    data_t *kptr = reinterpret_cast<data_t *>(params.d_k) + bh_id * params.k_bh_stride;
    data_t *vptr = reinterpret_cast<data_t *>(params.d_v) + bh_id * params.v_bh_stride + vdim_id * params.v_vdim_stride;
    data_t *qptr = reinterpret_cast<data_t *>(params.d_q) + bh_id * params.q_bh_stride;
    data_t *out = reinterpret_cast<data_t *>(params.d_out) + bh_id * params.out_bh_stride + vdim_id * params.out_vdim_stride;

    using namespace cute;
    constexpr int LOADATOM_PER_THREAD = 16/sizeof(data_t);
    cute::Tensor gQ = cute::make_tensor(cute::make_gmem_ptr(qptr), cute::make_shape(Int<chunk_size/LOADATOM_PER_THREAD>{},Int<LOADATOM_PER_THREAD>{}), cute::make_stride(Int<LOADATOM_PER_THREAD>{},Int<1>{}));
    using SmemLayoutQK = decltype(
        composition(
            Swizzle<2,2,3>{},// for 16itemsperthread, 32 banks, ld.shared.u128(?)
            Layout<Shape<Int<chunk_size/LOADATOM_PER_THREAD>,Int<LOADATOM_PER_THREAD>>,  // chunksize
                    Stride<Int<LOADATOM_PER_THREAD>,_1>>{}
        )
    );
    // using SmemLayoutQ = decltype(
    //     tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<NUM_THREADS>,Int<1>>{})
    // );
    // cute::Tensor sQ = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<data_t*>(smem)), cute::make_shape(params.l/LOADATOM_PER_THREAD,Int<LOADATOM_PER_THREAD>{}), cute::make_stride(Int<LOADATOM_PER_THREAD>{},Int<1>{}));
    cute::Tensor sQ = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<data_t*>(smem_q)), SmemLayoutQK{});
    using GmemCopyQK = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, data_t>{},
        // CACHEGLOBAL can only use uint128_t
            make_layout(Shape<Int<NUM_THREADS>,Int<1>>{}), // thread Layout
            make_layout(Shape<_1,Int<LOADATOM_PER_THREAD>>{})
        )
    );
    GmemCopyQK gmem_copy_qk;
    auto gmem_thr_copy_qk = gmem_copy_qk.get_thread_slice(threadIdx.x);
    Tensor gQ_partition = gmem_thr_copy_qk.partition_S(gQ);
    Tensor sQ_partition = gmem_thr_copy_qk.partition_D(sQ);

    using SmemCopyQK = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, data_t>{},
            make_layout(Shape<Int<NUM_THREADS>,Int<1>>{}), // thread Layout
            make_layout(Shape<_1,Int<ITEMS_PER_THREAD>>{}, Stride<Int<ITEMS_PER_THREAD>,_1>{})
        )
    );
    SmemCopyQK smem_copy_qk;
    using SmemLayoutQ1K1 = decltype(
        composition(
            Swizzle<2,2,3>{},// for 16itemsperthread, 32 banks, float32, ld.shared.u128(?)
            Layout<Shape<Int<chunk_size/ITEMS_PER_THREAD>,Int<ITEMS_PER_THREAD>>,  // chunksize
                    Stride<Int<ITEMS_PER_THREAD>,_1>>{}
        )
    );
    // cute::Tensor sQ1 = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<data_t*>(smem)), cute::make_shape(params.l/ITEMS_PER_THREAD,Int<ITEMS_PER_THREAD>{}), cute::make_stride(Int<ITEMS_PER_THREAD>{},Int<1>{}));
    cute::Tensor sQ1 = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<data_t*>(smem_q)), SmemLayoutQ1K1{});
    Tensor rQ = cute::make_tensor<data_t>(make_shape(_1{},Int<ITEMS_PER_THREAD>{}));
    auto smem_thr_copy_qk = smem_copy_qk.get_thread_slice(threadIdx.x);
    Tensor sQ1_copypartition = smem_thr_copy_qk.partition_S(sQ1);
    Tensor rQ_copypartition = smem_thr_copy_qk.partition_D(rQ);

    Tensor gK = cute::make_tensor(cute::make_gmem_ptr(kptr), cute::make_shape(Int<chunk_size/LOADATOM_PER_THREAD>{},Int<LOADATOM_PER_THREAD>{}), cute::make_stride(Int<LOADATOM_PER_THREAD>{},Int<1>{}));
    Tensor sK = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<data_t*>(smem_k)), SmemLayoutQK{});
    Tensor sK_partition = gmem_thr_copy_qk.partition_D(sK);
    Tensor gK_partition = gmem_thr_copy_qk.partition_S(gK);
    cute::Tensor sK1 = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<data_t*>(smem_k)), SmemLayoutQ1K1{});
    Tensor rK = cute::make_tensor<data_t>(make_shape(_1{},Int<ITEMS_PER_THREAD>{}));
    Tensor sK1_copypartition = smem_thr_copy_qk.partition_S(sK1);
    Tensor rK_copypartition = smem_thr_copy_qk.partition_D(rK);


    data_t data_decay[ITEMS_PER_THREAD];
    for (int chunk = 0; chunk < params.nchunk; chunk++) {
        Ktrait::BlockLoadT(smem_load).Load(decay, data_decay);//, params.l - chunk * params.chunk_size, 0.0f);

        data_t v_data[kNvdim][ITEMS_PER_THREAD];
        data_t out_data[kNvdim][ITEMS_PER_THREAD];
        // #pragma unroll
        for(int ii = 0; ii < kNvdim; ii++){
            for (int i = 0; i < ITEMS_PER_THREAD; i++) {
                out_data[ii][i] = 0.0f;
            }
        }
        #pragma unroll
        for(int ii = 0; ii < kNvdim; ii++){
            Ktrait::BlockLoadT(smem_load).Load(vptr, v_data[ii]);//, params.l - chunk * params.chunk_size, 0.0f);
            vptr += params.l;
            // __syncthreads();
        }
        /*
        // data_t q_data[ITEMS_PER_THREAD];
        // data_t k_data[ITEMS_PER_THREAD];
        // Ktrait::BlockLoadT(smem_load).Load(qptr, q_data);//, params.l - chunk * params.chunk_size, 0.0f);
        // Ktrait::BlockLoadT(smem_load).Load(kptr, k_data);//, params.l - chunk * params.chunk_size, 0.0f);
        for (int k = 0; k < params.block_kdim; k++) {
            data_t q_data[ITEMS_PER_THREAD];
            data_t k_data[ITEMS_PER_THREAD];
            Ktrait::BlockLoadT(smem_load).Load(kptr + k * params.l, k_data);//, params.l - chunk * params.chunk_size, 0.0f);
            scan_t thread_data[ITEMS_PER_THREAD];
            for (int i = 0; i < ITEMS_PER_THREAD; i++) {
                thread_data[i] = make_float2(data_decay[i], k_data[i] * v_data[i]);
            }
            // Compute exclusive prefix sum
            scan_t running_prefix;
            running_prefix =  chunk > 0 && threadIdx.x == 0? smem_running_prefix[k] : make_float2(0.f, 0.f);  
            BlockPrefixCallbackOp prefix_op(running_prefix);
            Ktrait::BlockScanT(smem_scan).InclusiveScan(thread_data, thread_data, ScanOp(), prefix_op);
            Ktrait::BlockLoadT(smem_load).Load(qptr + k * params.l, q_data);//, params.l - chunk * params.chunk_size, 0.0f);
            for (int i = 0; i < ITEMS_PER_THREAD; i++) {
                out_data[i] += thread_data[i].y * q_data[i];
            }
            __syncthreads();
            if (threadIdx.x == 0)
                smem_running_prefix[k] = prefix_op.running_prefix;
        }*/
        
        __syncthreads();
        // pipelined for 
        cute::copy(gmem_copy_qk, gK_partition, sK_partition);
        gK_partition.data() = gK_partition.data() + params.l;
        cute::cp_async_fence();
        cute::copy(gmem_copy_qk, gQ_partition, sQ_partition);
        gQ_partition.data() = gQ_partition.data() + params.l;
        cute::cp_async_fence();
        for (int k = 0; k < params.block_kdim-1; k++) {

            cute::cp_async_wait<1>();
            __syncthreads();
            cute::copy(smem_copy_qk, sK1_copypartition, rK_copypartition);
            Tensor k_data = rK;
            __syncthreads();

            cute::copy(gmem_copy_qk, gK_partition, sK_partition);
            gK_partition.data() = gK_partition.data() + params.l;
            cute::cp_async_fence();

            data_t thread_out_data[kNvdim][ITEMS_PER_THREAD];
            #pragma unroll
            for (int v = 0; v < kNvdim; v++){

                scan_t thread_data[ITEMS_PER_THREAD];
                for (int i = 0; i < ITEMS_PER_THREAD; i++) {
                    thread_data[i] = make_float2(data_decay[i], k_data[i] * v_data[v][i]);
                }
                // Compute exclusive prefix sum
                scan_t running_prefix;
                running_prefix =  chunk > 0 && threadIdx.x == 0? smem_running_prefix[k*kNvdim+v] : make_float2(0.f, 0.f);  
                BlockPrefixCallbackOp prefix_op(running_prefix);
                Ktrait::BlockScanT(smem_scan).InclusiveScan(thread_data, thread_data, ScanOp(), prefix_op);
                for (int i = 0; i < ITEMS_PER_THREAD; i++) {
                    thread_out_data[v][i] = thread_data[i].y;
                }
                __syncthreads();
                if (threadIdx.x == 0)
                    smem_running_prefix[k*kNvdim+v] = prefix_op.running_prefix;
            }
            
            cute::cp_async_wait<1>();
            __syncthreads();
            cute::copy(smem_copy_qk, sQ1_copypartition, rQ_copypartition);
            Tensor q_data = rQ;
            __syncthreads();

            cute::copy(gmem_copy_qk, gQ_partition, sQ_partition);
            gQ_partition.data() = gQ_partition.data() + params.l;
            cute::cp_async_fence();

            #pragma unroll
            for (int v = 0; v < kNvdim; v++){
                for (int i = 0; i < ITEMS_PER_THREAD; i++) {
                    out_data[v][i] += thread_out_data[v][i] * q_data[i];
                }
            }
        }
        {
            cute::cp_async_wait<1>();
            __syncthreads();
            cute::copy(smem_copy_qk, sK1_copypartition, rK_copypartition);
            Tensor k_data = rK;

            data_t thread_out_data[kNvdim][ITEMS_PER_THREAD];
            #pragma unroll
            for (int v = 0; v < kNvdim; v++){

                scan_t thread_data[ITEMS_PER_THREAD];
                for (int i = 0; i < ITEMS_PER_THREAD; i++) {
                    thread_data[i] = make_float2(data_decay[i], k_data[i] * v_data[v][i]);
                }
                // Compute exclusive prefix sum
                scan_t running_prefix;
                running_prefix =  chunk > 0 && threadIdx.x == 0? smem_running_prefix[(params.block_kdim-1)*kNvdim+v] : make_float2(0.f, 0.f);  
                BlockPrefixCallbackOp prefix_op(running_prefix);
                Ktrait::BlockScanT(smem_scan).InclusiveScan(thread_data, thread_data, ScanOp(), prefix_op);
                for (int i = 0; i < ITEMS_PER_THREAD; i++) {
                    thread_out_data[v][i] = thread_data[i].y;
                }
                __syncthreads();
                if (threadIdx.x == 0)
                    smem_running_prefix[(params.block_kdim-1)*kNvdim+v] = prefix_op.running_prefix;
            }
            cute::cp_async_wait<0>();
            __syncthreads();
            cute::copy(smem_copy_qk, sQ1_copypartition, rQ_copypartition);
            Tensor q_data = rQ;

            #pragma unroll
            for (int v = 0; v < kNvdim; v++){
                for (int i = 0; i < ITEMS_PER_THREAD; i++) {
                    out_data[v][i] += thread_out_data[v][i] * q_data[i];
                }
            }
        }
        
        __syncthreads();
        #pragma unroll
        for(int ii = 0; ii < kNvdim; ii++){
            Ktrait::BlockStoreT(smem_store).Store(out, out_data[ii]);
            out += params.l;
            // __syncthreads();
        }
        __syncthreads();

        // qptr += chunk_size;
        // gQ_partition.data() = qptr; // very slow and incorrect
        gQ_partition.data() = gQ_partition.data() + (-params.l*params.block_kdim) + chunk_size;
        // kptr += chunk_size;
        gK_partition.data() = gK_partition.data() + (-params.l*params.block_kdim) + chunk_size;
        vptr -= params.v_vdim_stride;
        vptr += chunk_size;
        out -= params.out_vdim_stride;
        out += chunk_size;
        decay += chunk_size;
    }
}


template <
    int                     BLOCK_THREADS_,
    int                     ITEMS_PER_THREAD_,
    BlockScanAlgorithm      ALGORITHM_,
    int                     kNvdim,
    typename data_t_, typename scan_t_>
void scan_fwd_launch(Params &params, cudaStream_t stream) {
    using Ktrait = scan_fwd_kernel_traits<BLOCK_THREADS_, ITEMS_PER_THREAD_, ALGORITHM_, kNvdim, data_t_, scan_t_>;
    constexpr int kSmemSize = Ktrait::kSmemSize + sizeof(scan_t) * MAX_KDIM * kNvdim;//MAX_VDIM;
    
    // printf("smem_size = %d\n", kSmemSize);  
    // printf("smem_load_size = %d\n", Ktrait::kSmemLoadSize);
    // printf("smem_store_size = %d\n", Ktrait::kSmemStoreSize);
    // printf("smem_scan_size = %d\n", Ktrait::kSmemScanSize);
    // dim3 grid(params.batch * params.heads, params.vdim/params.block_vdim, params.kdim/params.block_kdim);
    // dim3 grid(params.batch * params.heads, params.vdim/params.block_vdim, 1);
    assert(params.vdim % kNvdim == 0);
    dim3 grid(params.vdim/kNvdim, params.batch * params.heads, 1);
    // printf("grid = %d %d %d\n", grid.x, grid.y, grid.z);
    auto kernel = &scan_fwd_kernel<Ktrait>;
    if (kSmemSize >= 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
    }
    kernel<<<grid, Ktrait::BLOCK_THREADS, kSmemSize, stream>>>(params);
}



extern "C" void retnet_recurrent_fwd(void *d_q, void *d_k, void *d_v, void *d_decay, void *d_out, int batch, int heads, int kdim, int vdim, int l, int block_kdim) {
    Params params;
    params.d_q = d_q;
    params.d_k = d_k;
    params.d_v = d_v;
    params.d_decay = d_decay;
    params.d_out = d_out;
    params.batch = batch;
    params.heads = heads;
    params.vdim = vdim;
    params.kdim = kdim;
    params.l = l;
    params.block_kdim = block_kdim;
    // params.block_vdim = 1; // defined in kernel traits, because register(v_data, out_data) 正比于 vdim(vdim在循环最内层，增加load qk的data reuse)
    constexpr int block_vdim = 2;// 4;

    params.k_bh_stride = l * kdim;
    params.k_kdim_stride = block_kdim * l;

    params.v_bh_stride = l * vdim;
    params.v_vdim_stride = block_vdim * l;

    params.q_bh_stride = l * kdim;
    params.q_kdim_stride = block_kdim * l;

    params.out_bh_stride = l * vdim;
    params.out_vdim_stride = block_vdim * l;

    params.decay_head_stride = l;

    constexpr int Nthreads = 128;
    constexpr int Nitems = 16;
    constexpr int chunk_size = Nthreads * Nitems;
    params.nchunk = (l + chunk_size - 1) / chunk_size;

    scan_fwd_launch<Nthreads, Nitems, BLOCK_SCAN_WARP_SCANS,block_vdim, data_t, scan_t>(params, 0);
}
