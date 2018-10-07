#include <cudnn.h>
#include <stdlib.h>
#include <stdio.h>
#include "mat.cuh"

#define check(status) do{									\
	if(status!=0)											\
		{													\
			printf("cudnn returned none 0.\n");				\
			cudaDeviceReset();                              \
			exit(1);										\
		}													\
}while(0)

typedef struct convolution_parameters {
	int inChannel;
	int outChannel;
	int ksize;
	int height;
	int width;
	float alpha = 1;
	float beta = 0;
	int outSize;
	size_t workspaceSize;//workspace size needed
}CovParas;
class CovLayer {
public:
	CovLayer();
	int build(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int inChannel, int outChannel, int height, int width, int ksize);//build layer
	size_t buffer_size(void)
	{
		return paras.workspaceSize;
	}
	int load_para(FILE *fp);//copy paras from memory to GPU memory
	int ConvForward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void *x, void *workspace, size_t workspaceSize);//卷积
	int quantize_out(void *workspace);
	size_t get_result(void *out);//获取卷积结果
	int freeMem(void);//free data u and y, typically in reference
	int setMem(void);//set u and y
	//~CovLayer();

	CovParas paras;
	int step_w, step_y;//output quantization parameter
	cudnnTensorDescriptor_t uDesc, yDesc;//output descriptor
	void *u, *y;
	cudnnFilterDescriptor_t wDesc;//权重描述符
	cudnnTensorDescriptor_t bDesc;//偏置和输出描述符
	void *w, *b;

private:
	cudnnConvolutionDescriptor_t convDesc;//卷积描述符
	int algo_num;//number of avaliable algorithms
	cudnnConvolutionFwdAlgoPerf_t perfResults[8];//卷积算法描述符
};