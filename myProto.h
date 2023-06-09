#pragma once

#define DATASIZE 1000000
#define NUM_BINS 256
#define MASTER 0
#define OTHER_RANK 1
#define MY_TAG 0
#define NUM_BLOCKS 10
#define THREADS_PER_BLOCK 20
#define NUM_PROCESSES 2


int* generateRandomArray(int size);
void sendAndReceiveDataArray(int* dataArray, int dataSize, int** localData, int* localSize, int rank, int size);
void splitForCudaAndOmp(int** ompLocalData, int* ompLocalSize, int** cudaLocalData, int* cudaLocalSize, int* localData, int* localSize, int rank, int size);
// void computeHistogramParallelOMP(int* data, int dataSize, int** histogram);
void computeHistogramParallelOMP(int* data, int startIndex, int endIndex, int** histogram);
void computeHistogramParallelCUDA(int* data, int startIndex, int endIndex, int localSize, int** histogram);
// void reduceHistograms(int* localHistogramOMP, int* localHistogramCUDA, int** finalHistogram, int rank);
void buildBothHistogramArray(int* ompArray, int* cudaArray, int** bothArray);
void reduceHistograms(int* localHistogram, int** finalHistogram, int rank);
int* concatArrays(int* array1, int size1, int* array2, int size2);
void printHistogram(int* histogram, int size);