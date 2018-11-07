#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cudnn.h"

#define BLOCKSIZE 1024
#define GRIDSIZE 1024

#define ORI_FILE "..\\..\\data\\HEVC_Sequence\\BlowingBubbles_416x240_50.yuv"
#define INPUT_FILE "..\\..\\data\\BlowingBubbles\\BlowingBubbles_intra_main_HM16.7_anchor_416x240_10_Q%d.yuv"
//#define ORI_FILE "..\\..\\data\\HEVC_Sequence\\BasketballDrillText_832x480_50.yuv"
//#define INPUT_FILE "..\\..\\data\\BasketballDrillText\\BasketballDrillText_intra_main_HM16.7_anchor_832x480_10_Q%d.yuv"
//#define ORI_FILE "..\\..\\data\\HEVC_Sequence\\BQTerrace_1920x1080_60_10.yuv"
//#define INPUT_FILE "..\\..\\data\\BQTerrace\\BQTerrace_intra_main_HM16.7_anchor_1920x1080_10_Q%d.yuv"
//#define ORI_FILE "..\\..\\data\\HEVC_Sequence\\BQSquare_416x240_60.yuv"
//#define INPUT_FILE "..\\..\\data\\BQSquare\\BQSquare_intra_main_HM16.7_anchor_416x240_Q%d.yuv"
//#define ORI_FILE "..\\..\\data\\HEVC_Sequence\\FourPeople_1280x720_60.yuv"
//#define INPUT_FILE "..\\..\\data\\FourPeople\\FourPeople_intra_main_HM16.7_anchor_1280x720_10_Q%d.yuv"
//#define ORI_FILE "..\\..\\data\\HEVC_Sequence\\Traffic_2560x1600_30_crop.yuv"
//#define INPUT_FILE "..\\..\\data\\Traffic\\Traffic_intra_main_HM16.7_anchor_2560x1600_10_Q%d.yuv"
#define FRAME 1
#define CHANNEL 1
#define HEIGHT 240
#define WIDTH 416
#if HEIGHT*WIDTH*16/BLOCKSIZE >= 4096
#define MAXGRID 2048
#elif HEIGHT*WIDTH*16/BLOCKSIZE >= 2048
#define MAXGRID 1024
#elif HEIGHT*WIDTH*16/BLOCKSIZE >= 1024
#define MAXGRID 512
#elif HEIGHT*WIDTH*16/BLOCKSIZE >= 512
#define MAXGRID 256
#endif
//#define INT8_EXT_CONFIG
#define INT8x4_EXT_CONFIG
#ifdef INT8_EXT_CONFIG
#define XWFORMAT CUDNN_TENSOR_NHWC
#define XWTYPE CUDNN_DATA_INT8
#define YFORMAT CUDNN_TENSOR_NHWC
#define YTYPE CUDNN_DATA_FLOAT
#define BFORMAT CUDNN_TENSOR_NHWC
#define BTYPE CUDNN_DATA_INT32
#define CONVTYPE CUDNN_DATA_INT32
#define ALGO CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
typedef char xwtype;
typedef float convtype;//data type after convolution, bias and activation
typedef int btype;
typedef int midtype;//data type converted from uvtype
#define THRESHOLD 127
#endif
#ifdef INT8x4_EXT_CONFIG
#define XWFORMAT CUDNN_TENSOR_NCHW_VECT_C// Input and output features maps must be multiple of 4 
#define XWTYPE CUDNN_DATA_INT8x4
#define YFORMAT CUDNN_TENSOR_NCHW
#define YTYPE CUDNN_DATA_FLOAT
#define BFORMAT CUDNN_TENSOR_NCHW
#define BTYPE CUDNN_DATA_FLOAT
#define CONVTYPE CUDNN_DATA_INT32
#define ALGO CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
typedef char xwtype;
typedef float convtype;//data type after convolution, bias and activation
typedef float btype;
typedef int midtype;//data type converted from uvtype
#define THRESHOLD 127
#endif

__global__ void findMax_reduce1(convtype *g_idata, convtype *g_odata, int n);//n>=GRIDSIZE*BLOCKSIZE*2
__global__ void findMax_reduce2(convtype *data);//number = 1024
__host__ void findMax(convtype *data, convtype *buffer, int n, convtype *max);
__global__ void conv2mid(convtype *conv, midtype *mid, int num);
__global__ void VectorDiv(midtype *dividend, xwtype *quotient, int divisor, int n);
__global__ void applyRes(unsigned char *in, xwtype *res, unsigned char *recon);
char* HWCN2NCHW_VECT_C_CPU(char *HWCN, int H, int W, int C, int N, int *outSize);
char* NCHW2NCHW_VECT_C_CPU(char *NCHW, int N, int C, int H, int W, int *outSize);
__global__ void CHW2CHW_VECT_C(convtype *dividend, xwtype *quotient, int divisor, int C);
int NCHW2NCHW_VECT_C(convtype *dividend, xwtype *quotient, int divisor, int N, int C, int H, int W);
//void HWCN2NCHW_VECT_C(char *HWCN, char *NCHW, int H, int W, int C, int N);//C<=4 and expand C to 4
//void convdbg(xwtype *x, xwtype *w, convtype* u, btype *b);
