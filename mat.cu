#include <iostream>
#include "mat.cuh"
__global__ void applyRes(unsigned char *in, xwtype *res, unsigned char *recon)
{
	int i = blockDim.x*blockIdx.x+threadIdx.x;
	recon[i] = (int)in[i] + res[i];
}

__global__ void conv2mid(convtype *conv, midtype *mid, int num)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	while (i < num)
	{
		mid[i] = conv[i];
		i += gridDim.x*blockDim.x;
	}
}
__global__ void VectorDiv(midtype *dividend, xwtype *quotient, int divisor, int n)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	while (i < n)
	{
		quotient[i] = dividend[i] / divisor;
		i += gridDim.x*blockDim.x;
	}
}
__host__ void findMax(convtype *data, convtype *buffer, int n, convtype *max)
{
	findMax_reduce1 << <MAXGRID, BLOCKSIZE, BLOCKSIZE * sizeof(convtype) >> >(data, buffer, n);
	cudaDeviceSynchronize();
	findMax_reduce2 << <1, MAXGRID / 2, sizeof(convtype)*MAXGRID / 2 >> >(buffer);
	cudaDeviceSynchronize();
	cudaMemcpyAsync(max, buffer, sizeof(convtype), cudaMemcpyDeviceToHost);
}
__global__ void findMax_reduce1(convtype *g_idata, convtype *g_odata, int n)
{
	extern __shared__ convtype sdata[];//BLOCKSIZE>=blockDim.x
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
__global__ void findMax_reduce2(convtype *data)//number = 1024
{
	extern __shared__ convtype sdata[];
	unsigned int tid = threadIdx.x;
#if MAXGRID == 2048
	sdata[tid] = data[tid] > data[tid + 1024] ? data[tid] : data[tid + 1024];
#elif MAXGRID == 1024
	sdata[tid] = data[tid] > data[tid + 512] ? data[tid] : data[tid + 512];
#elif MAXGRID == 512
	sdata[tid] = data[tid] > data[tid + 256] ? data[tid] : data[tid + 256];
#elif MAXGRID == 256
	sdata[tid] = data[tid] > data[tid + 128] ? data[tid] : data[tid + 128];
#endif
#if MAXGRID >=2048
	if (tid < 512) { if (sdata[tid] < sdata[tid + 512]) sdata[tid] = sdata[tid + 512]; } __syncthreads();
#endif
#if MAXGRID >=1024
	if (tid < 256) { if (sdata[tid] < sdata[tid + 256]) sdata[tid] = sdata[tid + 256]; } __syncthreads();
#endif
#if MAXGRID >=512
	if (tid < 128) { if (sdata[tid] < sdata[tid + 128]) sdata[tid] = sdata[tid + 128]; } __syncthreads();
#endif
#if MAXGRID >=256
	if (tid < 64) { if (sdata[tid] < sdata[tid + 64]) sdata[tid] = sdata[tid + 64]; } __syncthreads();
#endif
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
char* HWCN2NCHW_VECT_C_CPU(char *HWCN, int H, int W, int C, int N, int *outSize)
{
	int i, j, k, m;
	int HWC_O, HW, HW4, W4, C_O, c_o, cv_o;
	HW = H*W;
	HW4 = HW * 4;
	W4 = W * 4;
	C_O = ceil((float)C / 4);
	HWC_O = H*W*C_O * 4;
	*outSize = N*C_O*H*W * 4;
	char *NCHW_VECT_C = new char[*outSize];
	memset(NCHW_VECT_C, 0, *outSize);
	for (i = 0;i < N;i++)
		for (j = 0;j < C;j++)
		{
			c_o = j >> 2;
			cv_o = j & 3;
			for (k = 0;k < H;k++)
				for (m = 0;m < W;m++)
					NCHW_VECT_C[i*HWC_O+c_o*HW4+k*W4+m*4+cv_o] = HWCN[k*W*C*N + m*C*N + j*N + i];
		}
	return NCHW_VECT_C;
}
char* NCHW2NCHW_VECT_C_CPU(char *NCHW, int N, int C, int H, int W, int *outSize)
{
	int i, j, k, m;
	int CHW, HW, HW4, W4, C_O, c_o, cv_o;
	CHW = C*H*W;
	HW = H*W;
	HW4 = HW * 4;
	W4 = W * 4;
	C_O = ceil((float)C / 4);
	*outSize = N*C_O*H*W * 4;
	char *NCHW_VECT_C = new char[*outSize];
	memset(NCHW_VECT_C, 0, *outSize);
	for (i = 0;i < N;i++)
		for (j = 0;j < C;j++)
		{
			c_o = j >> 2;
			cv_o = j & 3;
			for (k = 0;k < H;k++)
				for (m = 0;m < W;m++)
					NCHW_VECT_C[i*CHW + c_o*HW4 + k*W4 + m * 4 + cv_o] = NCHW[i*CHW + j*HW + k*W + m];
		}
	return NCHW_VECT_C;
}
__global__ void CHW2CHW_VECT_C(convtype *dividend, xwtype *quotient, int divisor, int C)//应明确指定用途，尤其是正负数
{
	int tid, gridSize;
	char j, c, cv;
	gridSize = gridDim.x*blockDim.x;
	tid = blockIdx.x*blockDim.x + threadIdx.x;
	for (j = 0;j < C;j++)
	{
		c = j >> 2;
		cv = j & 3;
		if(dividend[j*gridSize+tid]<0)
			quotient[(c*gridSize + tid) * 4 + cv] = (dividend[j*gridSize + tid] - (divisor >> 1)) / divisor;
		else
			quotient[(c*gridSize + tid) * 4 + cv] = (dividend[j*gridSize + tid] + (divisor >> 1)) / divisor;
	}
}

int NCHW2NCHW_VECT_C(convtype *dividend, xwtype *quotient, int divisor, int N, int C, int H, int W)
{
	int i;
	for (i = 0;i < N;i++)
		CHW2CHW_VECT_C <<<HEIGHT, WIDTH>>>(dividend + i*C*H*W, quotient + i*C*H*W, divisor, C);
	return 0;
}
void convdbg(xwtype *x, xwtype *w, convtype* u, btype *b)
{
	xwtype *x_h = new xwtype[HEIGHT*WIDTH * 4];
	xwtype *w_h = new xwtype[3 * 3 * 16];
	convtype *u_h = new convtype[HEIGHT*WIDTH*4];
	btype b_h[16];
	cudaMemcpy(x_h, x, sizeof(xwtype)*HEIGHT*WIDTH*4, cudaMemcpyDeviceToHost);
	cudaMemcpy(w_h, w, sizeof(xwtype)*3*3*16, cudaMemcpyDeviceToHost);
	cudaMemcpy(u_h, u, sizeof(convtype)*HEIGHT*WIDTH*4, cudaMemcpyDeviceToHost);
	cudaMemcpy(b_h, b, sizeof(convtype)*1, cudaMemcpyDeviceToHost);
	free(x_h);
	free(w_h);
	free(u_h);
	free(b_h);
}
