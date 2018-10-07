#include "mat.cuh"
__global__ void VectorAdd(cudnnType *a, cudnnType *b, cudnnType *c, cudnnType n)
{
	int i = threadIdx.x;
	if (i<n)
		c[i] = a[i] + b[i];
}
__global__ void VectorDiv(cudnnType *dividend, cudnnType *quotient, int divisor, int n)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	while (i < n)
	{
		quotient[i] = dividend[i] / divisor;
		i += gridDim.x*blockDim.x;
	}
}
__host__ void findMax(cudnnType *data, cudnnType *buffer, int n, cudnnType *max)
{
	findMax_reduce1 << <GRIDSIZE, BLOCKSIZE, BLOCKSIZE * sizeof(int) >> >(data, buffer, n);
	cudaDeviceSynchronize();
	findMax_reduce2 << <1, GRIDSIZE / 2, sizeof(int)*GRIDSIZE / 2 >> >(buffer);
	cudaDeviceSynchronize();
	cudaMemcpyAsync(max, buffer, sizeof(int), cudaMemcpyDeviceToHost);
}
__global__ void findMax_reduce1(cudnnType *g_idata, cudnnType *g_odata, int n)
{
	extern __shared__ cudnnType sdata[];//BLOCKSIZE>=blockDim.x
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;
	unsigned int gridSize = blockDim.x * gridDim.x;
	sdata[tid] = (abs(g_idata[i])>abs(g_idata[i + gridSize])) ? abs(g_idata[i]) : abs(g_idata[i + gridSize]);
	i += gridSize * 2;
	while (i < n) { if (sdata[tid] < abs(g_idata[i]))sdata[tid] = abs(g_idata[i]); i += gridSize; }
	__syncthreads();
	if (tid < 512) { if (sdata[tid] < sdata[tid + 512]) sdata[tid] = sdata[tid + 512]; }__syncthreads();
	if (tid < 256) { if (sdata[tid] < sdata[tid + 256]) sdata[tid] = sdata[tid + 256]; } __syncthreads();
	if (tid < 128) { if (sdata[tid] < sdata[tid + 128]) sdata[tid] = sdata[tid + 128]; } __syncthreads();
	if (tid < 64) { if (sdata[tid] < sdata[tid + 64]) sdata[tid] = sdata[tid + 64]; } __syncthreads();
	if (tid < 32)
	{
		if (sdata[tid] < sdata[tid + 32]) sdata[tid] = sdata[tid + 32];
		if (sdata[tid] < sdata[tid + 16]) sdata[tid] = sdata[tid + 16];
		if (sdata[tid] < sdata[tid + 8]) sdata[tid] = sdata[tid + 8];
		if (sdata[tid] < sdata[tid + 4]) sdata[tid] = sdata[tid + 4];
		if (sdata[tid] < sdata[tid + 2]) sdata[tid] = sdata[tid + 2];
		if (sdata[tid] < sdata[tid + 1]) sdata[tid] = sdata[tid + 1];
	}
	if (tid == 0) { g_odata[blockIdx.x] = sdata[0]; }
}
__global__ void findMax_reduce2(cudnnType *data)//number = 1024
{
	extern __shared__ cudnnType sdata[];
	unsigned int tid = threadIdx.x;
	sdata[tid] = data[tid] > data[tid + 512] ? data[tid] : data[tid + 512];
	if (tid < 256) { if (sdata[tid] < sdata[tid + 256]) sdata[tid] = sdata[tid + 256]; } __syncthreads();
	if (tid < 128) { if (sdata[tid] < sdata[tid + 128]) sdata[tid] = sdata[tid + 128]; } __syncthreads();
	if (tid < 64) { if (sdata[tid] < sdata[tid + 64]) sdata[tid] = sdata[tid + 64]; } __syncthreads();
	if (tid < 32)
	{
		if (sdata[tid] < sdata[tid + 32]) sdata[tid] = sdata[tid + 32];
		if (sdata[tid] < sdata[tid + 16]) sdata[tid] = sdata[tid + 16];
		if (sdata[tid] < sdata[tid + 8]) sdata[tid] = sdata[tid + 8];
		if (sdata[tid] < sdata[tid + 4]) sdata[tid] = sdata[tid + 4];
		if (sdata[tid] < sdata[tid + 2]) sdata[tid] = sdata[tid + 2];
		if (sdata[tid] < sdata[tid + 1]) sdata[tid] = sdata[tid + 1];
	}
	if (tid == 0)data[0] = sdata[tid];
}
