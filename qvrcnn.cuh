#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include "cnn.cuh"

#define LAYER 4
typedef struct {
	//w in decreasing order while b in increasing order
	int stepw[LAYER];
	int stepy[LAYER];
}step_parameters;
__global__ void adjustBasic(step_parameters steps, cudnnType*b, cudnnType *b_adj, int n);//scale n times
__global__ void adjustOutput(step_parameters steps, cudnnType*o, cudnnType *o_adj, int n);//scale n times
//quantized vrcnn
class qvrcnn{
public:
	qvrcnn(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int batch, int channel, int height, int width);//build qvrcnn
	int load(const char *filename);
	int forward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void *x);
	int backward();

private:
	int layers;
	CovLayer C1;
	CovLayer C2_1;
	CovLayer C2_2;
	CovLayer C3_1;
	CovLayer C3_2;
	CovLayer C4;
	int height, width;//feature map size
	step_parameters steps;
	void insert_w(int stepw, int layer);//layer means number of stepws
	void insert_y(int stepy, int layer);//layer means number of stepws
	cudnnTensorDescriptor_t conc1_Desc, conc2_Desc;//concate layer descriptor
	void *conc1, *conc2;
	int setConcat(cudnnTensorDescriptor_t conc_Desc, void *conc, int channel);
	int concat(CovLayer C1, CovLayer C2, void *p);
	void *workspace = nullptr;//workspace
	size_t workspaceSize;//工作空间

	//float* e; // 训练误差
	//float* L; // 瞬时误差能量
};
