#pragma once

#define DATASIZE 1000000
#define NUM_BINS 256
#define MASTER 0
#define NUM_BLOCKS 10
#define THREADS_PER_BLOCK 20
#define NUM_PROCESSES 2

int* generateRandomArray(int size);
void distributeDataArray(int* dataArray, int dataSize, int** localData, int* localSize);
void computeHistogramParallel(int* data, int dataSize, int** histogram);
void reduceHistograms(int* localHistogram, int localSize, int** finalHistogram);
void printHistogram(int* histogram, int size);