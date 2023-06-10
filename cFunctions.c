#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>

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


// /**
//  * Sends and receives data array between processes.
//  *
//  * @param dataArray The original data array.
//  * @param dataSize The size of the original data array.
//  * @param localData The local data array to be allocated and filled.
//  * @param localSize The size of the local data array.
//  * @param rank The rank of the current process.
//  * @param size The total number of processes.
//  */
// void sendAndReceiveDataArray(int* dataArray, int dataSize, int** localData, int* localSize, int rank, int size) 
// {
//     MPI_Status  status;
//     // Calculate the size of the local data array
//     if (rank ==  MASTER) // if doesnt divide in 2 give extra to master
//         (*localSize) = (DATASIZE + 1) / 2;
//     else
//         (*localSize) = DATASIZE / 2;
   
//     // Allocate memory for the local data array
//     *localData = (int*)malloc(*localSize * sizeof(int));
//     if (*localData == NULL) {
//         fprintf(stderr, "Error: Failed to allocate memory for localData.\n");
//         MPI_Abort(MPI_COMM_WORLD, 1);
//     }
//     if (rank == MASTER) 
//     {
//         // Send the first half of the data array to the other process
//         // MPI_Send(dataArray+(localSize), (DATASIZE / 2), MPI_INT, OTHER_RANK, MY_TAG, MPI_COMM_WORLD);
//         MPI_Send(&dataArray+((DATASIZE + 1) / 2), (DATASIZE / 2), MPI_INT, OTHER_RANK, MY_TAG, MPI_COMM_WORLD);
//     } 
//     else 
//     {
//         // Receive the first half of the data array from the master process
//         // MPI_Recv(localData, localSize, MPI_INT, MASTER, MY_TAG, MPI_COMM_WORLD, &status);
//         MPI_Recv(*localData, *localSize, MPI_INT, MASTER, MY_TAG, MPI_COMM_WORLD, &status);
//     }
// }

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
    MPI_Status status;

    // Calculate the size of the local data array
    if (rank == MASTER)
        (*localSize) = (DATASIZE + 1) / 2;
    else
        (*localSize) = DATASIZE / 2;

    // Allocate memory for the local data array
    *localData = (int*)malloc((*localSize) * sizeof(int));
    if (*localData == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for localData.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == MASTER) {
        // Send the first half of the data array to the other process
        MPI_Send(dataArray + ((*localSize)), DATASIZE / 2, MPI_INT, OTHER_RANK, MY_TAG, MPI_COMM_WORLD);
    } else {
        // Receive the first half of the data array from the master process
        MPI_Recv(*localData, (*localSize), MPI_INT, MASTER, MY_TAG, MPI_COMM_WORLD, &status);
    }
}



void computeHistogramParallelOMP(int* data, int startIndex, int endIndex, int** histogram) 
{
    printf("\nOOMMPP OLD start:%d NEW Start:%d, OLD end:%d NEW END:%d\n", startIndex, 0, endIndex, (endIndex - startIndex));
    // Allocate memory for the local histograsm
    *histogram = (int*)calloc(NUM_BINS, sizeof(int));
    if (*histogram == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for histogram.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // int i;
    // Perform OpenMP parallel processing on the local data array
    #pragma omp parallel for
        for (int i = 0; i < (endIndex - startIndex); i++) // omp runs from 0 to halfSize, cuda from halfSize to dataSize
            (*histogram)[data[i]] += 1;
}


void buildBothHistogramArray(int* ompArray, int* cudaArray, int** bothArray) {
    // Allocate memory for the local histograsm
    *bothArray = (int*)calloc(NUM_BINS, sizeof(int));
    if (*bothArray == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for histogram.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Add the OMP results to the temporary histogram
    for (int i = 0; i < NUM_BINS; i++) {
        (*bothArray)[i] += ompArray[i];
        (*bothArray)[i] += cudaArray[i];
        // printf("bothArray[%d]: %d\n",i, *bothArray[i]);
    }
}


/**
 * Reduces the local OpenMP and CUDA histograms to the final histogram on rank 0.
 *
 * @param localHistogramOMP Pointer to the local OpenMP histogram array
 * @param localHistogramCUDA Pointer to the local CUDA histogram array
 * @param finalHistogram Pointer to the final histogram array (allocated on rank 0)
 * @param rank The rank of the current process
 */
void reduceHistograms(int* localHistogram, int** finalHistogram, int rank)
{
    MPI_Status  status;

    // Allocate memory for the final histogram on rank 0
    if (rank == MASTER) 
    {
        int tempArray[NUM_BINS];
        MPI_Recv(tempArray, NUM_BINS, MPI_INT, OTHER_RANK, MY_TAG, MPI_COMM_WORLD, &status);
        buildBothHistogramArray(localHistogram, tempArray, finalHistogram);
        // *finalHistogram = (int*)calloc(NUM_BINS, sizeof(int));

    }
    else {
        MPI_Send(localHistogram, NUM_BINS, MPI_INT, MASTER, MY_TAG, MPI_COMM_WORLD);
    }

    // Reduce the local histogram to the final histogram on rank 0
    // MPI_Reduce(localHistogram, *finalHistogram, NUM_BINS, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
}


void printHistogram(int* histogram, int size) 
{
    printf("Histogram:\n");
    for (int i = 0; i < size; i++) 
    {
        printf("%d: %d\n", i, histogram[i]);
    }
}


// test if the total numbers in the hist queal to NUM_OF_ELEMENTS
void test(int *hist, int n, int rank, int type) {
    int i;
    int sum = 0;
    for (i = 0;   i < n;   i++) {
        sum += hist[i];
    }
    if(sum == DATASIZE)
        printf("\nRank:%d The test type:%d passed successfully %d\n", rank, type, sum); 
    else
        printf("\nRank:%d The test type:%d Failed %d\n", rank, type, sum); 

}
