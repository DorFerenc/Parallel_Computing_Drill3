#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "myProto.h"

int main(int argc, char* argv[]) {
    int rank, size;
    int* dataArray;
    int dataSize;
    int* localData;
    int localSize;
    int* localHistogram;
    int* finalHistogram;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Generate or read the data array
    if (rank == MASTER) {
        // Generate the large data array
        dataSize = DATASIZE;
        dataArray = generateRandomArray(dataSize);
    }

    // Distribute the data array to all processes
    distributeDataArray(dataArray, dataSize, &localData, &localSize);

    // Perform parallel processing using OpenMP and CUDA
    computeHistogramParallel(localData, localSize, &localHistogram);

    // Reduce the partial histograms to obtain the final histogram
    reduceHistograms(localHistogram, localSize, &finalHistogram);

    // Print the final histogram (rank MASTER process)
    if (rank == MASTER) {
        printHistogram(finalHistogram, NUM_BINS);
    }

    // Clean up
    free(dataArray);
    free(localData);
    free(localHistogram);
    free(finalHistogram);

    MPI_Finalize();
    return 0;
}
