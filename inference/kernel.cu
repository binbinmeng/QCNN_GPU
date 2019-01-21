#include <sstream>
#include <stdlib.h>
#include <windows.h>
#include <time.h>
#include "qvrcnn.cuh"

#define NCHW_VECT_C_QMODEL "D:\\zhaohengrui\\QCNN_GPU\\QCNN_GPU\\model\\HM16.0_blu\\qvrcnn_nchw_vect_c_8bit_qfp_%d.data"
#define HWCN_QMODEL "D:\\zhaohengrui\\QCNN_GPU\\QCNN_GPU\\model\\HM16.0_blu\\qvrcnn_hwcn_8bit_qfp_%d.data"
#define NCHW_MODEL "D:\\zhaohengrui\\QCNN_GPU\\QCNN_GPU\\model\\HM16.0\\vrcnn_nchw_%d.data"
#define HWCN_MODEL "D:\\zhaohengrui\\QCNN_GPU\\QCNN_GPU\\model\\HM16.0\\vrcnn_hwcn_%d.data"
#ifdef INT8x4_EXT_CONFIG
#define MODEL_NAME NCHW_VECT_C_QMODEL
#elif defined(FLOAT_CONFIG)
#define MODEL_NAME NCHW_MODEL
#endif // INT8x4_EXT_CONFIG

void convert_model(void)
{
	int i;
	for (i = 32;i < 33;i += 5)
	{
		model_qfp_HWCN2NCHW_VECT_C(HWCN_QMODEL, NCHW_VECT_C_QMODEL, i);
		//model_HWCN2NCHW_VECT_C(HWCN_QMODEL, NCHW_VECT_C_QMODEL, i);
		//model_HWCN2NCHW(HWCN_MODEL, NCHW_MODEL, i);
	}
}

void test_layer(char *model_fn, int frame, int channel, int height, int width, int outChannel, int ksize)
{
	//使用随机数据测试卷积时间，只需设置尺寸参数
	int num_gpus;
	FILE *fp;
	cudnnHandle_t cudnnHandle;
	InputLayer I1;
	CovLayer C1;
	void *workspace;
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;
	char ori_fn[] = "..\\..\\data\\HEVC_Sequence\\FourPeople_1280x720_60.yuv";
	char input_fn[] = "..\\..\\data\\anchor16.0\\PeopleOnStreet_intra_main_HM16.0_anchor_Q22.yuv";
	
	cudaGetDeviceCount(&num_gpus);
	cudaSetDevice(0);
	cudnnCreate(&cudnnHandle);
	vrcnn_data test_data(frame*channel, height, width);
	I1.build(frame, channel, height, width);
	C1.build(cudnnHandle, I1.xDesc, frame, height, width, channel, outChannel, ksize);
	test_data.read_data(ori_fn, input_fn);
	I1.load(test_data.input);
	I1.ppro();
	fopen_s(&fp, model_fn, "rb");
	C1.load_para(fp);
	fclose(fp);
	if(C1.workspaceSize>MAXGRID*sizeof(convtype))
		check(cudaMalloc(&workspace, C1.workspaceSize));
	else
		check(cudaMalloc(&workspace, MAXGRID*sizeof(convtype)));
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	//C1.ConvForward(cudnnHandle, I1.xDesc, I1.x_ppro, workspace, C1.workspaceSize);
	check(cudnnConvolutionForward(cudnnHandle, &C1.alpha, I1.xDesc,
		I1.x, C1.wDesc, C1.w, C1.convDesc,
		ALGO, workspace, C1.workspaceSize, &C1.beta,
		C1.uDesc, C1.u));
	cudaDeviceSynchronize();
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	//C1.quantize_out(workspace);
	//C1.viewmem();
	printf("time:%lldus",ElapsedMicroseconds.QuadPart);
}
void testqvrcnn(char *ori_fn, char *input_fn, char *model_fn, int frame, int channel, int height, int width)
{
	int i;
	int num_gpus;
	vrcnn_data test_data(frame, height, width);
	double psnr1, psnr2;
	time_t now;
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;
	FILE*logfile;

	cudaGetDeviceCount(&num_gpus);
	qvrcnn qvrcnn1(0, 1, channel, height, width);//GPU_num,NCHW
	qvrcnn1.load_static_para(model_fn);
	test_data.read_data(ori_fn, input_fn);
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	for (i = 0;i < frame;i++)
	{
		qvrcnn1.load_data(test_data.input + i*channel*height*width);
		qvrcnn1.forward_blu();
		cudaDeviceSynchronize();
		cudaMemcpy(test_data.recon + i*channel*height*width, (datatype*)qvrcnn1.I1.x_rec, channel*height*width, cudaMemcpyDeviceToHost);
	}
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	now = time(0);
	//test_data.save_recon_as("recon.yuv");
	//test_data.psnr_pf();
	psnr1 = test_data.psnr(test_data.input);
	psnr2 = test_data.psnr(test_data.recon);
	printf("\nbefore net:PSNR=%.3f\nafter quantized net:PSNR=%.3f\ntime:%lldus\n", psnr1, psnr2, ElapsedMicroseconds.QuadPart);
	if (fopen_s(&logfile, "log.txt", "a+"))
		printf("write file failed\n");
	fprintf(logfile, "\nQVRCNN test date:%sdata:%s\nframes:%d\nheight:%d\nwidth:%d\nbefore net:PSNR=%f\nafter quantized net:PSNR=%f\ntime:%lldus\n", ctime(&now), input_fn, frame, height, width, psnr1, psnr2, ElapsedMicroseconds.QuadPart);
	fclose(logfile);
	if (fopen_s(&logfile, "recon_psnr.data", "ab+"))
		printf("open psnr file failed\n");
	fwrite(&psnr2, sizeof(double), 1, logfile);
	fclose(logfile);
}
int run_all(char*oriname,char*inputname,int height,int width)
{
	int qp;
	char input_fn[200];
	char model_fn[200];
	for (qp = 22; qp < 23; qp += 5)
	{
		sprintf_s(input_fn, "%sQ%d.yuv", inputname, qp);
		sprintf_s(model_fn, MODEL_NAME, qp);
		testqvrcnn(oriname, input_fn, model_fn, FRAME, CHANNEL, height, width);
		cudaDeviceSynchronize();
		//test_layer(model_fn, FRAME, 64, 720, 1280, 64, 5);
	}
	return 0;
}
int main(int argc, char**argv)
{
	//run_all(ORI_FILE, INPUT_FILE, HEIGHT, WIDTH);
	run_all(argv[1],argv[2],atoi(argv[3]),atoi(argv[4]));
	//convert_model();
	//system("pause");
}
