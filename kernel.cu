#include <sstream>
#include <stdlib.h>
#include <time.h>
#include "qvrcnn.cuh"
//HWCN2NHWC
#define HWCN_QMODEL "model\\qvrcnn_ppro_hwcn_8bit_%d.data"
#define NHWC_VECT_QMODEL "model\\qvrcnn_ppro_nhwc_vect_8bit_%d.data"

void convert_model(void)
{
	model_HWCN2NCHW_VECT_C(HWCN_QMODEL, NHWC_VECT_QMODEL, 22);
	model_HWCN2NCHW_VECT_C(HWCN_QMODEL, NHWC_VECT_QMODEL, 32);
	model_HWCN2NCHW_VECT_C(HWCN_QMODEL, NHWC_VECT_QMODEL, 37);
}

void test_layer(char *ori_fn, char *input_fn, char *model_fn, int frame, int height, int width)
{
	int num_gpus;
	FILE *fp;
	vrcnn_data test_data(frame, height, width);
	cudnnHandle_t cudnnHandle;
	cudnnTensorDescriptor_t xDesc;
	void *x, *x_h, *workspace;
	int xSize;
	
	cudaGetDeviceCount(&num_gpus);
	cudaSetDevice(0);
	cudnnCreate(&cudnnHandle);
	cudnnCreateTensorDescriptor(&xDesc);
	if (XWFORMAT == CUDNN_TENSOR_NCHW_VECT_C)
		cudnnSetTensor4dDescriptor(xDesc, XWFORMAT, XWTYPE, 1, 4, 240, 416);
	else
		cudnnSetTensor4dDescriptor(xDesc, XWFORMAT, XWTYPE, 1, 1, 240, 416);
	CovLayer C1;
	C1.build(cudnnHandle, xDesc, frame, height, width, 1, 64, 5);

	fopen_s(&fp, model_fn, "rb");
	C1.load_para(fp);
	//memdbg((convtype*)C1.u, (xwtype*)C1.v, (btype*)C1.b, C1.uSize);
	fclose(fp);
	if(C1.workspaceSize>MAXGRID*sizeof(convtype))
		cudaMalloc(&workspace, C1.workspaceSize);
	else
		cudaMalloc(&workspace, MAXGRID*sizeof(convtype));
	test_data.read_data(ori_fn, input_fn);
	test_data.preprocess();

	x_h = NCHW2NCHW_VECT_C_CPU(test_data.norm, frame, 1, height, width, &xSize);
	cudaMalloc(&x, sizeof(restype)*xSize);
	cudaMemcpy(x, x_h, sizeof(restype)*xSize, cudaMemcpyHostToDevice);
	free(x_h);
	
	C1.ConvForward(cudnnHandle, xDesc, x, workspace, C1.workspaceSize);
	C1.quantize_out(workspace);
	//C1.viewmem();
}
void testqvrcnn(char *ori_fn, char *input_fn, char *model_fn, int frame, int channel, int height, int width)
{
	int i, j;
	int num_gpus;
	vrcnn_data test_data(frame, height, width);
	float psnr1, psnr2;
	clock_t start_t, end_t;
	double total_t;

	cudaGetDeviceCount(&num_gpus);
	qvrcnn qvrcnn1(0, 1, channel, height, width);
	qvrcnn1.load_para(model_fn);
	test_data.read_data(ori_fn, input_fn);
	start_t = clock();
	for (i = 0;i < frame;i++)
	{
		qvrcnn1.load_data(test_data.input + i*channel*height*width);
		qvrcnn1.forward();
		cudaDeviceSynchronize();
		cudaMemcpy(test_data.recon + i*channel*height*width, (datatype*)qvrcnn1.I1.x_rec, channel*height*width, cudaMemcpyDeviceToHost);
	}
	end_t = clock();
	total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	test_data.psnr_pf();
	psnr1 = test_data.psnr(test_data.input);
	psnr2 = test_data.psnr(test_data.recon);
	printf("\nbefore net:PSNR=%f\nafter quantized net:PSNR=%f\ntime:%f\n", psnr1, psnr2, total_t);
}
int run_all()
{
	int qp;

	char ori_fn[200];
	char input_fn[200];
	char model_fn[200];

	for (qp = 37; qp < 38; qp += 5)
	{
		sprintf_s(ori_fn, ORI_FILE);
		sprintf_s(input_fn, INPUT_FILE, qp);
		sprintf_s(model_fn, NHWC_VECT_QMODEL, qp);
		testqvrcnn(ori_fn, input_fn, model_fn, FRAME, CHANNEL, HEIGHT, WIDTH);
		//testLayer(orifile, input, FRAME, HEIGHT, WIDTH, qp);
	}
	return 0;
}
int main(void)
{
	run_all();
	//convert_model();
}
