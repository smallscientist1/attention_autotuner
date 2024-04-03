// /usr/local/cuda/bin/nvcc test_hardware.cu -o test_hardware
#include <cstdio>
#include<cuda_runtime.h>
#include<stdio.h>

int main(){
    int dev=0,driverVersion=0,runtimeVersion=0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("device: %s\n",deviceProp.name);
    printf("SMEM per SM: %lu bytes\n",deviceProp.sharedMemPerMultiprocessor);
    printf("max SMEM per block: %lu bytes\ngmem bus width: %d bits\n",\
    deviceProp.sharedMemPerBlock,\
    deviceProp.memoryBusWidth);
    printf("cuda capability Major/Minor: %d.%d\n",deviceProp.major,deviceProp.minor);
    printf("register per block: %d\n", deviceProp.regsPerBlock);
}