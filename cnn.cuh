#include <cudnn.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mat.cuh"
#include "yuv_data.h"

#define check(status) do{									\
	if(status!=0)											\
		{													\
			printf("cudnn returned none 0.\n");				\
			cudaDeviceReset();                              \
			exit(1);										\
		}													\
}while(0)

class CovLayer {
public:
	CovLayer();
	int build(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int batch, int height, int width, int inChannel, int outChannel, int ksize);//build layer
	int load_para(FILE *fp);//copy paras from memory to GPU memory
	int ConvForward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void *x, void *workspace, size_t workspaceSize);//¾í»ı
	int activate(cudnnHandle_t cudnnHandle);
	int quantize_out(void* workspace);//u(float)->mid(int)->y(char)
	int viewmem(xwtype*x);
	int freeMem(void);//free data u and y, typically in reference
	int setMem(void);//set u and y
	~CovLayer();

	int batch;
	int height;
	int width;
	int inChannel;
	int outChannel;
	int ksize;
	int wSize, uSize, vSize;
	size_t workspaceSize;//workspace size needed
	int max_u;
	int step_w, step_y;//output quantization parameter
	float alpha = 1;
	float beta = 0;
	cudnnFilterDescriptor_t wDesc;//È¨ÖØÃèÊö·û
	cudnnTensorDescriptor_t bDesc;//Æ«ÖÃºÍÊä³öÃèÊö·û
	void *w, *b, *b_adj;
	cudnnTensorDescriptor_t uDesc, vDesc;//output descriptor
	void *u, *v;
	cudnnConvolutionDescriptor_t convDesc;//¾í»ıÃèÊö·û
	cudnnActivationDescriptor_t actiDesc;
	//int algo_num;//number of avaliable algorithms
	//cudnnConvolutionFwdAlgoPerf_t perfResults[8];//¾í»ıËã·¨ÃèÊö·û
};
class ConcatLayer {
public:
	ConcatLayer(void);
	int build(int batch, int height, int width, int inChannel1, int inChannel2);
	int concat(CovLayer *C1, CovLayer *C2, void *workspace);
	~ConcatLayer();
	int batch;
	int height;
	int width;
	int inChannel1;
	int inChannel2;
	int outChannel;
	cudnnTensorDescriptor_t concDesc;
	void *conc;
};
__global__ void HW2HW_VECT_C_PPRO(datatype*x, xwtype*x_ppro);
class InputLayer {
public:
	InputLayer(void);
	int build(int batch, int channel, int height, int width);
	int load(datatype *input);
	int ppro(void);
	int applyRes(xwtype*res);
	~InputLayer(void);
	
	int batch;
	int height;
	int width;
	int inChannel;
	int outChannel;
	int inSize, outSize;
	cudnnTensorDescriptor_t xDesc;
	void *x, *x_ppro, *x_rec;
};
