#pragma once

// #include <stdio.h>
// #include <stdlib.h>
// // #include <mpi.h>
// #include <omp.h>
// // #include <cstring>
// #include <string.h>

#define DATASIZE 1000
#define NUM_BINS 256
#define MASTER 0
#define OTHER_RANK 1
#define MY_TAG 0
#define NUM_BLOCKS 10
#define THREADS_PER_BLOCK 20
#define NUM_PROCESSES 2


int* generateRandomArray(int size);
void sendAndReceiveDataArray(int* dataArray, int dataSize, int** localData, int* localSize, int rank, int size);
void computeHistogramParallelOMP(int* data, int startIndex, int endIndex, int** histogram);
void computeOnGPU(int* data, int startIndex, int endIndex, int localSize, int** histogram);
void buildBothHistogramArray(int* ompArray, int* cudaArray, int** bothArray);
void reduceHistograms(int* localHistogram, int** finalHistogram, int rank);
void printHistogram(int* histogram, int size);
void test(int *hist, int n, int rank, int type);