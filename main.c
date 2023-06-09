#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
// #include <cstring>

#include "myProto.h"

int main(int argc, char* argv[]) {
   int rank, size;
   int* dataArray;
   int* localData;
   int localSize;
   int* localHistogramOMP;
   int* localHistogramCUDA;
   int* localHistogramBOTH;
   int* finalHistogram;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Status  status;

   if (size != 2) {
      perror("Run the example with two processes only\n");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);



   if (rank ==  MASTER) // if doesnt divide in 2 give extra to master
      localSize = (DATASIZE + 1) / 2;
   else
      localSize = DATASIZE / 2;
   
   // Allocate memory for the local data array
   localData = (int*)malloc(localSize * sizeof(int));
   if (localData == NULL) {
      fprintf(stderr, "Error: Failed to allocate memory for localData.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
   }

   if (rank == MASTER) {
      // Generate the large data array
      dataArray = generateRandomArray(DATASIZE);
      MPI_Send(dataArray+(localSize), (DATASIZE / 2), MPI_INT, OTHER_RANK, MY_TAG, MPI_COMM_WORLD);
      localData = dataArray;
   }
   else 
      MPI_Recv(localData, localSize, MPI_INT, MASTER, MY_TAG, MPI_COMM_WORLD, &status);



   // // Distribute the data array to all processes
   // sendAndReceiveDataArray(dataArray, DATASIZE, &localData, &localSize, rank, size);

   // Split the data into two halves for omp take the smaller half if there is a reminder (bigger will be in cuda)
   // Perform parallel processing using OpenMP and CUDA
   int startIndex = rank * (localSize + (DATASIZE % 2));
   int endIndex = rank * localSize + (localSize / 2 + (DATASIZE % 2)); //localSize + ((localSize + 1) / 2);
   computeHistogramParallelOMP(localData, startIndex, endIndex, &localHistogramOMP);
   printf("Done OMP: %d\n", rank);
   computeOnGPU(localData, endIndex, ((localSize + (DATASIZE % 2)) * rank) + localSize, localSize, &localHistogramCUDA);
   printf("Done CUDA22: %d\n", rank);

      test(localHistogramOMP, NUM_BINS, rank, 0);
      test(localHistogramCUDA, NUM_BINS, rank, 1);

   buildBothHistogramArray(localHistogramOMP, localHistogramCUDA, &localHistogramBOTH);
   printf("Done BOTH: %d\n", rank);  
   // if (rank == MASTER)
   //    test(localHistogramBOTH, NUM_BINS, rank, 3);
   //    printHistogram(localHistogramBOTH, NUM_BINS);
   // Reduce the partial histograms to obtain the final histogram
   reduceHistograms(localHistogramBOTH, &finalHistogram, rank);

   // Print the final histogram (rank MASTER process)
   if (rank == MASTER) {
      // test and print the results
      test(finalHistogram, NUM_BINS, rank, 4);
      // printHistogram(finalHistogram, NUM_BINS);
   }

   // Clean up
   free(dataArray);
   free(localData);
   free(localHistogramOMP);
   free(localHistogramCUDA);
   free(localHistogramBOTH);
   free(finalHistogram);

   MPI_Finalize();
   return 0;
}
