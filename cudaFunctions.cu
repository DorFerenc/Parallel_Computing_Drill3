#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"
#include <stdio.h>


__global__ void computeHistogramCUDA(int* data, int dataSize, int* histogram) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (tid < dataSize) {
        atomicAdd(&histogram[data[tid]], 1);
        tid += stride;
    }
}

void computeHistogramParallelCUDA(int* data, int dataSize, int** histogram) {
    int* cudaData;
    int* cudaHistogram;

    // Allocate device memory for data and histogram
    cudaMalloc((void**)&cudaData, dataSize * sizeof(int));
    cudaMalloc((void**)&cudaHistogram, NUM_BINS * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(cudaData, data, dataSize * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize histogram on device memory
    cudaMemset(cudaHistogram, 0, NUM_BINS * sizeof(int));

    // Launch kernel for parallel histogram computation
    computeHistogramCUDA<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(cudaData, dataSize, cudaHistogram);

    // Copy histogram result from device to host
    *histogram = (int*)malloc(NUM_BINS * sizeof(int));
    cudaMemcpy(*histogram, cudaHistogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up device memory
    cudaFree(cudaData);
    cudaFree(cudaHistogram);
}