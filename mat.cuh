#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cudnn.h"

#define BLOCKSIZE 1024
#define GRIDSIZE 1024
#define INT8_QUANTIZE

#ifdef INT8_QUANTIZE
#define CUDNNTYPE CUDNN_DATA_INT32
typedef int cudnnType;
#define THRESHOLD 127
#endif

__global__ void findMax_reduce1(cudnnType *g_idata, cudnnType *g_odata, int n);//n>=GRIDSIZE*BLOCKSIZE*2
__global__ void findMax_reduce2(cudnnType *data);//number = 1024
__host__ void findMax(cudnnType *data, cudnnType *buffer, int n, cudnnType *max);
__global__ void VectorDiv(cudnnType *dividend, cudnnType *quotient, int divisor, int n);
