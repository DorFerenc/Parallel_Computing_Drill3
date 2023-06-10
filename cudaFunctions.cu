#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"
#include <stdio.h>


// __global__ void computeHistogramCUDA(int* data, int dataSize, int* histogram) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     // int chunck = (dataSize / (NUM_BLOCKS * THREADS_PER_BLOCK));
//     int stride = gridDim.x * blockDim.x;

//     for (int i = tid; i < dataSize; i += stride) {
//         atomicAdd(&histogram[data[i]], 1);
//     }
// }

__global__ void computeHistogramCUDA(int* data, int startIndex, int endIndex, int* histogram) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid + startIndex; i < endIndex; i += stride) {
        atomicAdd(&histogram[data[i]], 1);
    }
}

__global__  void initHist(int* h) {

  int index = threadIdx.x;
  h[index] = 0;

}

void computeOnGPU(int* data, int startIndex, int endIndex, int localSize, int** histogram) {

    printf("\nCCUUDDAA start:%d, NEW START:%d, end:%d NEW END:%d\n", startIndex, (endIndex - startIndex) + (localSize % 2), endIndex, localSize);
    startIndex = endIndex - startIndex + (localSize % 2);
    endIndex = localSize;

     // Allocate device memory on GPU for CUDA data and histogram from Host (CPU)
    cudaError_t cudaStatus;
    int* cudaData = NULL;
    int* cudaHistogram = NULL;

    cudaStatus = cudaMalloc((void**)&cudaData, localSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for cudaData: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }

    cudaStatus = cudaMalloc((void**)&cudaHistogram, NUM_BINS * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for cudaHistogram: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }

    // Copy data from host to device (GPU memory)
    cudaStatus = cudaMemcpy(cudaData, data, localSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed from host to device: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }

    // Initialize histogram on device memory 
    initHist<<<1, NUM_BINS>>>(cudaHistogram);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA memset failed for cudaHistogram: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }

    // Launch kernel for parallel histogram computation
    // computeHistogramCUDA<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(cudaData, localSize, cudaHistogram);
    computeHistogramCUDA<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(cudaData, startIndex, endIndex, cudaHistogram);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }

    // // Synchronize to ensure all CUDA operations are completed
    // cudaDeviceSynchronize();


    // Copy histogram result from device (GPU) to host (CPU)
    *histogram = (int*)malloc(NUM_BINS * sizeof(int));
    cudaStatus = cudaMemcpy(*histogram, cudaHistogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed from device to host: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }

    // Clean up device memory
    if (cudaFree(cudaData) != cudaSuccess || cudaFree(cudaHistogram) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }

}