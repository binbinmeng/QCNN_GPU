#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include "cnn.cuh"

//#define MEM_DBG
#define LAYER 4
typedef struct {
	//w in decending order while y in increasing order
	int stepw[LAYER];
	int stepy[LAYER];
}step_parameters;

int model_HWCN2NCHW_VECT_C(const char* filein, const char *fileout, int qp);
int model_HWCN2NCHW(const char* filein, const char *fileout, int qp);
int model_NCWH2HWCN(const char* filein, const char *fileout, int qp);
int model_qfp_HWCN2NCHW_VECT_C(const char* filein, const char *fileout, int qp);
__global__ void adjustBasic(step_parameters steps, btype*b, btype *b_adj, int n);//scale n times
__global__ void adjustOutput(step_parameters steps, convtype*o, xwtype *o_adj, int n, int num, int gridSize);//scale n times
__global__ void adjustOutput_static(convtype*o, xwtype *o_adj, int multiplier, int shifts, int num, int gridSize);//scale n times
//quantized vrcnn
class qvrcnn{
public:
	qvrcnn(int gpu_num, int batch, int channel, int height, int width);//build qvrcnn
	int load_para(char *filename);
	int load_static_para(char *filename);
	int load_data(datatype *input);
	int forward(void);
	int forward_blu(void);
	int backward(void);
	int save_b_adj(const char*filename);
	~qvrcnn();

//private:
	int layers;
	cudnnHandle_t cudnnHandle;
	
	InputLayer I1;
	CovLayer C1;
	CovLayer C2_1;
	CovLayer C2_2;
	ConcatLayer Conc1;
	CovLayer C3_1;
	CovLayer C3_2;
	ConcatLayer Conc2;
	CovLayer C4;
	int batch, channel, height, width;//input size
	step_parameters steps;
	void insert_w(int stepw, int layer);//layer means number of stepws
	void insert_y(int stepy, int layer);//layer means number of stepws
	void *workspace = nullptr;//workspace
	size_t workspaceSize;//工作空间

	//float* e; // 训练误差
	//float* L; // 瞬时误差能量
};
