#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "myProto.h"

int* generateRandomArray(int size) {
    int* array = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        array[i] = rand() % NUM_BINS;
    }
    return array;
}

void distributeDataArray(int* dataArray, int dataSize, int** localData, int* localSize) {
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate the size of the local data array
    int quotient = dataSize / size;
    int remainder = dataSize % size;
    *localSize = quotient;
    if (rank < remainder) {
        (*localSize)++;
    }

    // Allocate memory for the local data array
    *localData = (int*)malloc(*localSize * sizeof(int));

    // Determine the displacement and counts for MPI_Scatterv
    int* sendCounts = (int*)malloc(size * sizeof(int));
    int* displacements = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        sendCounts[i] = quotient;
        if (i < remainder) {
            sendCounts[i]++;
        }
        displacements[i] = i > 0 ? displacements[i - 1] + sendCounts[i - 1] : 0;
    }

    // Scatter the data array to all processes
    MPI_Scatterv(dataArray, sendCounts, displacements, MPI_INT, *localData, *localSize, MPI_INT, 0, MPI_COMM_WORLD);

    // Clean up
    free(sendCounts);
    free(displacements);
}

void computeHistogramParallel(int* data, int dataSize, int** histogram) {
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Allocate memory for the local histogram
    *histogram = (int*)calloc(NUM_BINS, sizeof(int));

    // Perform OpenMP parallel processing on the local data array
    #pragma omp parallel for
    for (int i = 0; i < dataSize; i++) {
        #pragma omp atomic
        (*histogram)[data[i]]++;
    }
}

void reduceHistograms(int* localHistogram, int localSize, int** finalHistogram) {
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate memory for the final histogram on rank 0
    if (rank == MASTER) {
        *finalHistogram = (int*)calloc(NUM_BINS, sizeof(int));
    }

    // Reduce the local histograms to the final histogram on rank 0
    MPI_Reduce(localHistogram, *finalHistogram, NUM_BINS, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
}

void printHistogram(int* histogram, int size) {
    printf("Histogram:\n");
    for (int i = 0; i < size; i++) {
        printf("%d: %d\n", i, histogram[i]);
    }
}
