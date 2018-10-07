#include <stdio.h>
#include <sstream>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include "yuv_data.h"

struct qvrcnn_data {
	char weight[5*5*64];
	char bias[64];
};//实验性，仅读取第一层

struct qvrcnn_data* read_qvrcnn(void)
{
	struct qvrcnn_data* net_data = new struct qvrcnn_data;
	FILE  *fp = NULL;
	if (fopen_s(&fp, "model\\qvrcnn_8bit_22.data", "rb"))
		printf("open file failed\n");
		fread(net_data->weight, sizeof(char), 5*5*64, fp);
	fclose(fp);
	return net_data;
}

int main(int argc, char** argv)
{
	cublasHandle_t cublasHandle;
	cudnnHandle_t cudnnHandle;
	cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor;//数据和偏置描述符
	cudnnFilterDescriptor_t conv1filterDesc;//权重描述符
	cudnnConvolutionDescriptor_t conv1Desc;//卷积描述符
	//cudnnConvolutionFwdAlgo_t conv1algo;//卷积算法描述符
	cudnnConvolutionFwdAlgoPerf_t perfResults[8];
	size_t sizeInBytes;

	YChannel *ydata, *conv1_out;
	qvrcnn_data* net_data;
	int batch = 1, channel = 0, height = 240, width = 416, return_value, return_value1;
	std::stringstream filename;
	filename << "data\\BlowingBubbles_intra_main_HM16.7_anchor_416x240_10_Q22.yuv";
	ydata = get_Y(filename.str().c_str(), batch, height, width);
	conv1_out = get_Y(NULL, 64, height, width);

	int num_gpus;
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
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_INT32,
		batch, 64, 1, 1);

	return_value = cudnnSetTensor4dDescriptor(dataTensor,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_INT32,
		1, 1, height, width);
	return_value = cudnnSetFilter4dDescriptor(conv1filterDesc,
		CUDNN_DATA_INT32,
		CUDNN_TENSOR_NCHW,
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
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_INT32,
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

	//读取或初始化网络权重
	net_data = read_qvrcnn();

	char *d_data;
	int *d_conv1;//前向传播数据
	float alpha = 1.0f, beta = 0.0f;
	return_value = cudaMalloc(&d_data, sizeof(char) *1*416*240);//在GPU中分配空间
	return_value = cudaMalloc(&d_conv1, sizeof(char) *64*416*240);

	char *d_pconv1, *d_pconv1bias;//网络参数
	return_value = cudaMalloc(&d_pconv1, sizeof(char) * 5*5*64);
	return_value = cudaMalloc(&d_pconv1bias, sizeof(char) * 64);
	
	void *d_cudnn_workspace = nullptr;//缓存和工作空间
	if (sizeInBytes > 0)
		return_value = cudaMalloc(&d_cudnn_workspace, sizeInBytes);//分配工作空间

	return_value = cudaMemcpyAsync(d_pconv1, net_data->weight, sizeof(char) * 5*5*64, cudaMemcpyHostToDevice);//拷贝网络到GPU
	return_value = cudaDeviceSynchronize();//同步GPU
	return_value = cudaMemcpyAsync(d_data, ydata->ImgData,
		sizeof(char) * ydata->frames*ydata->h*ydata->w, cudaMemcpyHostToDevice);//拷贝数据到GPU

	return_value = cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor,
		d_data, conv1filterDesc, d_pconv1, conv1Desc,
		perfResults[0].algo, d_cudnn_workspace, sizeInBytes, &beta,
		conv1Tensor, d_conv1);//进行一次卷积运算
	return_value = cudaDeviceSynchronize();//同步GPU

	return_value = cudaMemcpy(conv1_out->ImgData, d_conv1, sizeof(char) * 64 * 416 * 240, cudaMemcpyDeviceToHost);
	//到此步即可完成debug						
	//cudaMemcpy(&conv1.pconv[0], d_pconv1, sizeof(float) * conv1.pconv.size(), cudaMemcpyDeviceToHost);//从GPU中拷贝出数据
	return_value = cudaFree(d_data);//释放内存
	return_value = cudaFree(d_conv1);
	return_value = cudaFree(d_pconv1);
	return_value = cudaFree(d_pconv1bias);
	return_value = cudaFree(d_cudnn_workspace);

	return 0;
}