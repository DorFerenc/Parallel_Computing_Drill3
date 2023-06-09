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

    // for (int i = tid * chunck; i < tid * chunck + chunck; i++) {
    //     atomicAdd(&histogram[data[i]], 1);
    // }
    // while (tid < dataSize) {
    //     atomicAdd(&histogram[data[tid]], 1);
    //     tid += chunck;
    // }
// }

// __global__  void buildHist(int *h, int *temp) {
//     int index = threadIdx.x;
    

//     for (int i=0; i < NUM_BLOCKS; i++)
//         for(int j=0; j< THREADS_PER_BLOCK; j++)
//             h[index] += temp[index + (j * NUM_BINS) + (NUM_BINS * THREADS_PER_BLOCK * i)];
// }

__global__  void initHist(int* h) {

  int index = threadIdx.x;
  h[index] = 0;

}

void computeOnGPU(int* data, int startIndex, int endIndex, int localSize, int** histogram) {

    printf("\nCCUUDDAA start:%d, end:%d\n", startIndex, endIndex);
    int temp = startIndex;
    startIndex = endIndex - startIndex;
    endIndex = localSize;
    printf("\nCCUUDDAA NEW start:%d, end:%d\n", startIndex, endIndex);
    //  // Split the data into two halves for omp take the smaller half if there is a reminder (bigger will be in cuda)
    // int cudaDataSize  = dataSize / 2;
    // int remainder = dataSize % 2;
    // if (remainder > 0) { // if doesnt divide in 2 give extra to cuda
    //     cudaDataSize ++;
    // }

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
    // cudaStatus = cudaMemset(cudaHistogram, 0, NUM_BINS * sizeof(int));
    // if (cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "CUDA memset failed for cudaHistogram: %s\n", cudaGetErrorString(cudaStatus));
    //     exit(EXIT_FAILURE);
    // }

    // Launch kernel for parallel histogram computation
    // computeHistogramCUDA<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(cudaData, localSize, cudaHistogram);
    computeHistogramCUDA<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(cudaData, startIndex, endIndex, cudaHistogram);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }


    // Synchronize to ensure all CUDA operations are completed
    cudaDeviceSynchronize();


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


    int result = 0;
    for (int i = 0;  i < NUM_BINS;   i++)
        result += (*histogram)[i];
    if (result == DATASIZE / 4)
        printf("Test PASSED\n");
    else
        printf("Test FAILED!!!!! %d\n", result);

    printf("Done Cuda\n");

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