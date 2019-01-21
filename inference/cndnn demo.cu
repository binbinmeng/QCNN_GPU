/*
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include "yuv_data.h"

struct qvrcnn_data {
	char weight[5 * 5 * 64];
	int bias[64];
};//实验性，仅读取第一层

void HWCN2NHWC(char *HWCN, char *NHWC, int H, int W, int C, int N)
{
	int i, j, k, m;
	for (i = 0;i < H;i++)
		for (j = 0;j < W;j++)
			for (k = 0;k < C;k++)
				for (m = 0;m < N;m++)
					NHWC[m*H*W*C + i*W*C + j*C + k] = HWCN[i*W*C*N + j*C*N + k*N + m];
}

struct qvrcnn_data* read_qvrcnn(void)
{
	struct qvrcnn_data net_data_HWCN;
	struct qvrcnn_data* net_data_NHWC = new struct qvrcnn_data;
	FILE  *fp = NULL;
	if (fopen_s(&fp, "model\\qvrcnn_ppro_8bit_27.data", "rb"))
		printf("open file failed\n");
	fseek(fp, sizeof(int), SEEK_CUR);
	fread(net_data_HWCN.weight, sizeof(char), 5 * 5 * 64, fp);
	fread(net_data_NHWC->bias, sizeof(int), 64, fp);
	fclose(fp);
	HWCN2NHWC(net_data_HWCN.weight, net_data_NHWC->weight, 5, 5, 1, 64);//convert format
	return net_data_NHWC;
}

int main(int argc, char** argv)
{
	int num_gpus;
	cublasHandle_t cublasHandle;
	cudnnHandle_t cudnnHandle;
	cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor;//数据和偏置描述符
	cudnnFilterDescriptor_t conv1filterDesc;//权重描述符
	cudnnConvolutionDescriptor_t conv1Desc;//卷积描述符
	cudnnConvolutionFwdAlgoPerf_t perfResults[8];
	size_t sizeInBytes;

	YChannel *ydata;
	Res *ydata_reg;
	qvrcnn_data* net_data;
	int batch = 1, channel = 0, height = 240, width = 416, return_value, return_value1;
	std::stringstream filename;

	cudaGetDeviceCount(&num_gpus);
	cudaSetDevice(0);
	cublasCreate(&cublasHandle);
	cudnnCreate(&cudnnHandle);
	cudnnCreateTensorDescriptor(&dataTensor);//初始化张量描述符
	cudnnCreateTensorDescriptor(&conv1Tensor);
	cudnnCreateTensorDescriptor(&conv1BiasTensor);

	cudnnCreateFilterDescriptor(&conv1filterDesc);//初始化权重描述符

	cudnnCreateConvolutionDescriptor(&conv1Desc);//初始化卷积描述符

												 //设置卷积描述符
	return_value = cudnnSetTensor4dDescriptor(conv1BiasTensor,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_INT32,
		1, 64, 1, 1);
	
	return_value = cudnnSetTensor4dDescriptor(dataTensor,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_INT8,
		1, 1, height, width);
	return_value = cudnnSetFilter4dDescriptor(conv1filterDesc,
		CUDNN_DATA_INT8,
		CUDNN_TENSOR_NHWC,
		64, 1, 5, 5);
	return_value = cudnnSetConvolution2dDescriptor(conv1Desc,
		2, 2,
		1, 1,
		1, 1,
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_INT32);
	return_value = cudnnGetConvolution2dForwardOutputDim(conv1Desc,
		dataTensor,
		conv1filterDesc,
		&batch, &channel, &height, &width);
	return_value = cudnnSetTensor4dDescriptor(conv1Tensor,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1, 64, height, width);
	cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &return_value);
	return_value = cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle,
		dataTensor,
		conv1filterDesc,
		conv1Desc,
		conv1Tensor,
		8,
		&return_value1,
		perfResults);
	return_value = cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
		dataTensor,
		conv1filterDesc,
		conv1Desc,
		conv1Tensor,
		perfResults[0].algo,
		&sizeInBytes);

	char *x;
	float *conv,*conv_h;//前向传播数据
	char *w;
	int *b;//网络参数
	void *d_cudnn_workspace = nullptr;//缓存和工作空间
	float alpha = 1.0f, beta = 0.0f;
	clock_t start_t, end_t;
	double total_t;
	int i;
	//读取网络和数据
	net_data = read_qvrcnn();
	filename << "data\\BlowingBubbles_intra_main_HM16.7_anchor_416x240_10_Q27.yuv";
	ydata = get_Y(filename.str().c_str(), batch, height, width);
	ydata_reg = regularize(ydata);

	return_value = cudaMalloc(&x, sizeof(char) * 416 * 240);//在GPU中分配空间
	return_value = cudaMalloc(&w, sizeof(char) * 5 * 5 * 64);
	return_value = cudaMalloc(&b, sizeof(int) * 64);
	return_value = cudaMalloc(&conv, sizeof(float) * 64 * 416 * 240);
	conv_h = (float*)malloc(sizeof(float) * 64 * 416 * 240);
	if (sizeInBytes > 0)
		return_value = cudaMalloc(&d_cudnn_workspace, sizeInBytes);//分配工作空间

	return_value = cudaMemcpyAsync(w, net_data->weight, sizeof(char) * 5 * 5 * 64, cudaMemcpyHostToDevice);//拷贝网络到GPU
	return_value = cudaMemcpyAsync(x, ydata_reg->data,sizeof(char) * ydata->frames*ydata->h*ydata->w, cudaMemcpyHostToDevice);//拷贝数据到GPU

	start_t = clock();
	for (i = 0;i < 10000;i++)
	{
		return_value = cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor,
			x, conv1filterDesc, w, conv1Desc,
			perfResults[0].algo, d_cudnn_workspace, sizeInBytes, &beta,
			conv1Tensor, conv);//进行一次卷积运算
		return_value = cudaDeviceSynchronize();//同步GPU
	}
	end_t = clock();
	total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	printf("%f", total_t);
	return_value = cudaMemcpy(conv_h, conv, sizeof(float) * 64 * 416 * 240, cudaMemcpyDeviceToHost);
	//到此步即可完成debug						
	//cudaMemcpy(&conv1.pconv[0], d_pconv1, sizeof(float) * conv1.pconv.size(), cudaMemcpyDeviceToHost);//从GPU中拷贝出数据
	return_value = cudaFree(x);//释放内存
	return_value = cudaFree(w);
	return_value = cudaFree(b);
	return_value = cudaFree(conv);
	return_value = cudaFree(d_cudnn_workspace);
	system("pause");
	return 0;
}
*/
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include "yuv_data.h"

struct vrcnn_data {
	float weight[5 * 5 * 64];
	float bias[64];
};//实验性，仅读取第一层

void HWCN2NHWC(float *HWCN, float *NHWC, int H, int W, int C, int N)
{
	int i, j, k, m;
	for (i = 0;i < H;i++)
		for (j = 0;j < W;j++)
			for (k = 0;k < C;k++)
				for (m = 0;m < N;m++)
					NHWC[m*H*W*C + i*W*C + j*C + k] = HWCN[i*W*C*N + j*C*N + k*N + m];
}

struct vrcnn_data* read_vrcnn(void)
{
	struct vrcnn_data net_data_HWCN;
	struct vrcnn_data* net_data_NHWC = new struct vrcnn_data;
	FILE  *fp = NULL;
	if (fopen_s(&fp, "model\\vrcnn_ppro_27.data", "rb"))
		printf("open file failed\n");
	fread(net_data_HWCN.weight, sizeof(float), 5 * 5 * 64, fp);
	fread(net_data_NHWC->bias, sizeof(float), 64, fp);
	fclose(fp);
	HWCN2NHWC(net_data_HWCN.weight, net_data_NHWC->weight, 5, 5, 1, 64);//convert format
	return net_data_NHWC;
}
float *regularizef(YChannel *ydata)
{
	int i;
	float *reg = (float*)malloc(sizeof(float)*ydata->frames*ydata->h*ydata->w);
	for (i = 0; i < ydata->frames*ydata->h*ydata->w; i++)
		reg[i] = (int)ydata->ImgData[i] - 128;
	return reg;

}
int main(int argc, char** argv)
{
	int num_gpus;
	cublasHandle_t cublasHandle;
	cudnnHandle_t cudnnHandle;
	cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor;//数据和偏置描述符
	cudnnFilterDescriptor_t conv1filterDesc;//权重描述符
	cudnnConvolutionDescriptor_t conv1Desc;//卷积描述符
	cudnnConvolutionFwdAlgoPerf_t perfResults[8];
	size_t sizeInBytes;

	YChannel *ydata;
	float *ydata_reg;
	vrcnn_data* net_data;
	int batch = 1, channel = 0, height = 240, width = 416, return_value, return_value1;
	std::stringstream filename;

	cudaGetDeviceCount(&num_gpus);
	cudaSetDevice(0);
	cublasCreate(&cublasHandle);
	cudnnCreate(&cudnnHandle);
	cudnnCreateTensorDescriptor(&dataTensor);//初始化张量描述符
	cudnnCreateTensorDescriptor(&conv1Tensor);
	cudnnCreateTensorDescriptor(&conv1BiasTensor);

	cudnnCreateFilterDescriptor(&conv1filterDesc);//初始化权重描述符

	cudnnCreateConvolutionDescriptor(&conv1Desc);//初始化卷积描述符

												 //设置卷积描述符
	return_value = cudnnSetTensor4dDescriptor(conv1BiasTensor,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1, 64, 1, 1);

	return_value = cudnnSetTensor4dDescriptor(dataTensor,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1, 1, height, width);
	return_value = cudnnSetFilter4dDescriptor(conv1filterDesc,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NHWC,
		64, 1, 5, 5);
	return_value = cudnnSetConvolution2dDescriptor(conv1Desc,
		2, 2,
		1, 1,
		1, 1,
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_FLOAT);
	return_value = cudnnGetConvolution2dForwardOutputDim(conv1Desc,
		dataTensor,
		conv1filterDesc,
		&batch, &channel, &height, &width);
	return_value = cudnnSetTensor4dDescriptor(conv1Tensor,
		CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		1, 64, height, width);
	cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &return_value);
	return_value = cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle,
		dataTensor,
		conv1filterDesc,
		conv1Desc,
		conv1Tensor,
		8,
		&return_value1,
		perfResults);
	return_value = cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
		dataTensor,
		conv1filterDesc,
		conv1Desc,
		conv1Tensor,
		perfResults[0].algo,
		&sizeInBytes);

	char *x;
	float *conv, *conv_h;//前向传播数据
	float *w;
	int *b;//网络参数
	void *d_cudnn_workspace = nullptr;//缓存和工作空间
	float alpha = 1.0f, beta = 0.0f;
	clock_t start_t, end_t;
	double total_t;
	int i;
	//读取网络和数据
	net_data = read_vrcnn();
	filename << "data\\BlowingBubbles_intra_main_HM16.7_anchor_416x240_10_Q27.yuv";
	ydata = get_Y(filename.str().c_str(), batch, height, width);
	ydata_reg = regularizef(ydata);

	return_value = cudaMalloc(&x, sizeof(float) * 416 * 240);//在GPU中分配空间
	return_value = cudaMalloc(&w, sizeof(float) * 5 * 5 * 64);
	return_value = cudaMalloc(&b, sizeof(float) * 64);
	return_value = cudaMalloc(&conv, sizeof(float) * 64 * 416 * 240);
	conv_h = (float*)malloc(sizeof(float) * 64 * 416 * 240);
	if (sizeInBytes > 0)
		return_value = cudaMalloc(&d_cudnn_workspace, sizeInBytes);//分配工作空间

	return_value = cudaMemcpyAsync(w, net_data->weight, sizeof(float) * 5 * 5 * 64, cudaMemcpyHostToDevice);//拷贝网络到GPU
	return_value = cudaMemcpyAsync(x, ydata_reg, sizeof(float) * ydata->frames*ydata->h*ydata->w, cudaMemcpyHostToDevice);//拷贝数据到GPU

	start_t = clock();
	for (i = 0;i < 10000;i++)
	{
		return_value = cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor,
			x, conv1filterDesc, w, conv1Desc,
			perfResults[0].algo, d_cudnn_workspace, sizeInBytes, &beta,
			conv1Tensor, conv);//进行一次卷积运算
		return_value = cudaDeviceSynchronize();//同步GPU
	}
	end_t = clock();
	total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	printf("%f", total_t);
	return_value = cudaMemcpy(conv_h, conv, sizeof(float) * 64 * 416 * 240, cudaMemcpyDeviceToHost);
	//到此步即可完成debug						
	//cudaMemcpy(&conv1.pconv[0], d_pconv1, sizeof(float) * conv1.pconv.size(), cudaMemcpyDeviceToHost);//从GPU中拷贝出数据
	return_value = cudaFree(x);//释放内存
	return_value = cudaFree(w);
	return_value = cudaFree(b);
	return_value = cudaFree(conv);
	return_value = cudaFree(d_cudnn_workspace);
	system("pause");
	return 0;
}