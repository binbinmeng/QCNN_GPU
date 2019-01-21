/*
YUV data input and output
*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "mat.cuh"
typedef unsigned char datatype; //此处适用于8bit像素位宽
typedef char restype;
class vrcnn_data {
public:
	vrcnn_data(int frame, int height, int width);
	int read_data(char *orifile, char *inputfile);
	int read_frame(char *orifile, char *inputfile, int n);
	int preprocess(void);//已改为在GPU中处理
	int loadRes_GPU(xwtype*v);
	int applyRes(void);
	double psnr(datatype*data);
	double psnr_pf(void);
	int save_recon_as(char* filename);
	~vrcnn_data(void);

	int frame, h, w, nSize, xSize;
	datatype *ori, *input, *recon;
	restype *norm, *res;
};
