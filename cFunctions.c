#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "myProto.h"

/**
 * Generates a random array of integers.
 *
 * @param size The size of the array to generate.
 * @return A pointer to the generated array.
 */
int* generateRandomArray(int size) {
    int* array = (int*)malloc(size * sizeof(int));
    if (array == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for the array.\n");
        return NULL;
    }
    
    for (int i = 0; i < size; i++) {
        array[i] = rand() % NUM_BINS;
    }
    
    return array;
}


/**
 * Sends and receives data array between processes.
 *
 * @param dataArray The original data array.
 * @param dataSize The size of the original data array.
 * @param localData The local data array to be allocated and filled.
 * @param localSize The size of the local data array.
 * @param rank The rank of the current process.
 * @param size The total number of processes.
 */
void sendAndReceiveDataArray(int* dataArray, int dataSize, int** localData, int* localSize, int rank, int size) 
{
    // Calculate the size of the local data array
    int quotient = dataSize / 2;
    int remainder = dataSize % 2;
    *localSize = quotient;
    if (rank < remainder) { // if doesnt divide in 2 give extra to master
        (*localSize)++;
    }

    // Allocate memory for the local data array
    *localData = (int*)malloc(*localSize * sizeof(int));
    if (*localData == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for localData.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == MASTER) 
    {
        // Send the first half of the data array to the other process
        MPI_Send(&dataArray+(*localSize), *localSize, MPI_INT, OTHER_RANK, MY_TAG, MPI_COMM_WORLD);
    } 
    else 
    {
        // Receive the first half of the data array from the master process
        MPI_Recv(*localData, *localSize, MPI_INT, MASTER, MY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

/**
 * Splits the local data array into separate arrays for OpenMP and CUDA computations.
 * 
 * @param ompLocalData Pointer to the array to store the data for OpenMP computation.
 * @param ompLocalSize Pointer to the variable to store the size of the OpenMP data array.
 * @param cudaLocalData Pointer to the array to store the data for CUDA computation.
 * @param cudaLocalSize Pointer to the variable to store the size of the CUDA data array.
 * @param localData Pointer to the original local data array.
 * @param localSize Pointer to the variable storing the size of the local data array.
 * @param rank Rank of the current process.
 * @param size Total number of processes.
 */
void splitForCudaAndOmp(int** ompLocalData, int* ompLocalSize, int** cudaLocalData, int* cudaLocalSize, int* localData, int* localSize, int rank, int size) {
    // Calculate the size of the local data array
    int quotient = *localSize / 2;
    int remainder = *localSize % 2;
    *ompLocalSize = quotient;
    if (remainder > 0) { // if doesnt divide in 2 give extra to omp
        (*ompLocalSize)++;
    }
    *cudaLocalSize = *localSize - *ompLocalSize;

    // Allocate memory for the arrays
    *ompLocalData = (int*)malloc(*ompLocalSize * sizeof(int));
    *cudaLocalData = (int*)malloc(*cudaLocalSize * sizeof(int));

    // Check for successful memory allocation
    if (*ompLocalData == NULL || *cudaLocalData == NULL) {
        printf("Memory allocation failed for ompLocalData or cudaLocalData.\n");
        exit(1);
    }

    // Split the local data into ompLocalData and cudaLocalData
    int splitIndex = *ompLocalSize;

    // Copy data for OpenMP computation
    memcpy(*ompLocalData, localData, splitIndex * sizeof(int));

    // Copy data for CUDA computation
    memcpy(*cudaLocalData, localData + splitIndex, (*cudaLocalSize) * sizeof(int));
}


void computeHistogramParallelOMP(int* data, int dataSize, int** histogram) 
{
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Allocate memory for the local histogram
    *histogram = (int*)calloc(NUM_BINS, sizeof(int));

    // Perform OpenMP parallel processing on the local data array
    #pragma omp parallel for
    for (int i = 0; i < dataSize; i++) 
    {
        #pragma omp atomic
        (*histogram)[data[i]]++;
    }
}

void reduceHistograms(int* localHistogram, int localSize, int** finalHistogram) 
{
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate memory for the final histogram on rank 0
    if (rank == MASTER) 
    {
        *finalHistogram = (int*)calloc(NUM_BINS, sizeof(int));
    }

    // Reduce the local histograms to the final histogram on rank 0
    MPI_Reduce(localHistogram, *finalHistogram, NUM_BINS, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
}

void printHistogram(int* histogram, int size) 
{
    printf("Histogram:\n");
    for (int i = 0; i < size; i++) 
    {
        printf("%d: %d\n", i, histogram[i]);
    }
}
