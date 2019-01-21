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
	int load_para(FILE *fp);//copy paras from file to GPU memory
	int load_static_para(FILE *fp);//copy paras from file to GPU memory
	int ConvForward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void *x, void *workspace, size_t workspaceSize);//¾í»ý
	int ConvForward_static(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void *x, void *workspace, size_t workspaceSize);//¾í»ý
	int activate(cudnnHandle_t cudnnHandle);
#if defined(INT8_EXT_CONFIG)||defined(INT8x4_EXT_CONFIG)
	int quantize_out(void* workspace);//u(float)->mid(int)->y(char)
	int quantize_out_fix(int max_u);//u(float)->mid(int)->y(char)
	int quantize_out_static(void);
	int quantize_out_blu(void);//u(float)->v(char)
#endif
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
	//dynamic params
	int max_u;
	int step_w, step_y;//output quantization parameter
	//static params
	int blu;
	int mul, shift;
	float alpha = 1;
	float beta = 0;
	cudnnFilterDescriptor_t wDesc;//È¨ÖØÃèÊö·û
	cudnnTensorDescriptor_t bDesc;//Æ«ÖÃºÍÊä³öÃèÊö·û
	void *w, *b, *b_adj;
	cudnnTensorDescriptor_t uDesc, vDesc;//output descriptor
	void *u, *v;
	cudnnConvolutionDescriptor_t convDesc;//¾í»ýÃèÊö·û
	cudnnActivationDescriptor_t actiDesc;
	//int algo_num;//number of avaliable algorithms
	//cudnnConvolutionFwdAlgoPerf_t perfResults[8];//¾í»ýËã·¨ÃèÊö·û
};
class ConcatLayer {
public:
	ConcatLayer(void);
	int build(int batch, int height, int width, int inChannel1, int inChannel2);
	int concat(CovLayer *C1, CovLayer *C2, void *workspace);
	int concat_fix(CovLayer *C1, CovLayer *C2, convtype max1, convtype max2);
	int concat_static(CovLayer *C1, CovLayer *C2);
	int ConcatLayer::concat_blu(CovLayer *C1, CovLayer *C2);
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
class InputLayer {
public:
	InputLayer(void);
	int build(int batch, int channel, int height, int width);
	int load(datatype *input);
	int ppro(void);
	int applyRes(xwtype*res);
	int applyRes_y(convtype*res, int mul, int shift);
	int viewmem(xwtype*res);
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
