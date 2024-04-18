#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <thrust/device_vector.h>
#include <vector>

float do_bench(std::function<void()> k, int ms_warmup=50, int ms_measure=100 ) { // void (*k)() wrong for [&] () {}

    thrust::device_vector<int> cache(int(256e6/4));

  float ms;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  k();
  cudaEventRecord(start, 0);
  for (int _ = 0; _ < 5; _++) {
    k();
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  int warm_up = int(ceil(ms_warmup / (ms/5)));
  int measure = int(ceil(ms_measure / (ms/5)));
    for (int _ = 0; _ < warm_up; _++) {
        k();
    }

    std::vector<cudaEvent_t> start_(measure), stop_(measure);
    for (int i = 0; i < measure; i++) {
        cudaEventCreate(&start_[i]);
        cudaEventCreate(&stop_[i]);
    }
    for(int i = 0; i < measure; i++) {
        thrust::fill(cache.begin(), cache.end(), i);
        cudaEventRecord(start_[i], 0);
        k();
        cudaEventRecord(stop_[i], 0);
    }
    if(cudaDeviceSynchronize()!=cudaSuccess) {
        printf("CUDA error\n");
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    }
    std::vector<float> ms_(measure);
    for(int i = 0; i < measure; i++) {
        cudaEventElapsedTime(&ms_[i], start_[i], stop_[i]);
    }
    ms = 0;
    for(int i=0;i<measure;i++) {
        ms += ms_[i];
    }
    ms /= measure;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < measure; i++) {
        cudaEventDestroy(start_[i]);
        cudaEventDestroy(stop_[i]);
    }

    return ms;
}
