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
void computeHistogramParallel(int* data, int dataSize, int** histogram);
void reduceHistograms(int* localHistogram, int localSize, int** finalHistogram);
void printHistogram(int* histogram, int size);