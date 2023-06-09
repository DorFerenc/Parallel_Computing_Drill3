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

void computeHistogramParallelCUDA(int* data, int startIndex, int endIndex, int localSize, int** histogram) {
    //  // Split the data into two halves for omp take the smaller half if there is a reminder (bigger will be in cuda)
    // int cudaDataSize  = dataSize / 2;
    // int remainder = dataSize % 2;
    // if (remainder > 0) { // if doesnt divide in 2 give extra to cuda
    //     cudaDataSize ++;
    // }

     // Allocate device memory on GPU for CUDA data and histogram from Host (CPU)
    cudaError_t cudaStatus;
    int* cudaData;
    int* cudaHistogram;

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

    //TODO Initialize arrays on device initHist <<< 1 , RANGE >>> (d_h);

    // Initialize histogram on device memory 
    cudaStatus = cudaMemset(cudaHistogram, 0, NUM_BINS * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA memset failed for cudaHistogram: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }

    // Launch kernel for parallel histogram computation
    computeHistogramCUDA<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(cudaData, localSize, cudaHistogram);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }

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


    // // Allocate device memory for data and histogram
    // cudaMalloc((void**)&cudaData, halfSize * sizeof(int));
    // cudaMalloc((void**)&cudaHistogram, NUM_BINS * sizeof(int));

    // // Copy data from host to device
    // cudaMemcpy(cudaData, data, halfSize * sizeof(int), cudaMemcpyHostToDevice);

    // // Initialize histogram on device memory
    // cudaMemset(cudaHistogram, 0, NUM_BINS * sizeof(int));

    // // Launch kernel for parallel histogram computation
    // computeHistogramCUDA<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(cudaData, dataSize, cudaHistogram);

    // // Copy histogram result from device to host
    // *histogram = (int*)malloc(NUM_BINS * sizeof(int));
    // cudaMemcpy(*histogram, cudaHistogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // // Clean up device memory
    // cudaFree(cudaData);
    // cudaFree(cudaHistogram);
}