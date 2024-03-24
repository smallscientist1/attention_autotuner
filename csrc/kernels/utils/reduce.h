#pragma once

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
template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void eight_allreduce_(cute::Tensor<Engine0, Layout0> &dst, cute::Tensor<Engine1, Layout1> &src, Operator &op) {
    using namespace cute;
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<8>::run(src(i), op);
    }
}
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void reduce_(cute::Tensor<Engine0, Layout0> const& tensor, cute::Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    // quad_allreduce_(summary, summary, op);
    eight_allreduce_(summary, summary, op);
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
