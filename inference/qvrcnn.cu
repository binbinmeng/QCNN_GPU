#include "qvrcnn.cuh"
#include <stdio.h>

qvrcnn::qvrcnn(int gpu_num, int batch, int channel, int height, int width)//build qvrcnn
{
	check(cudaSetDevice(gpu_num));
	check(cudnnCreate(&cudnnHandle));
	this->batch = batch, this->channel = channel, this->height = height, this->width = width;
	
	I1.build(batch, channel, height, width);
	C1.build(cudnnHandle, I1.xDesc, batch, height, width, I1.outChannel, 64, 5);
	C2_1.build(cudnnHandle, C1.vDesc, batch, height, width, C1.outChannel, 32, 3);
	C2_2.build(cudnnHandle, C1.vDesc, batch, height, width, C1.outChannel, 16, 5);
	Conc1.build(batch, height, width, C2_1.outChannel, C2_2.outChannel);
	C3_1.build(cudnnHandle, Conc1.concDesc, batch, height, width, Conc1.outChannel, 16, 3);
	C3_2.build(cudnnHandle, Conc1.concDesc, batch, height, width, Conc1.outChannel, 32, 1);
	Conc2.build(batch, height, width, C3_1.outChannel, C3_2.outChannel);
	C4.build(cudnnHandle, Conc2.concDesc, batch, height, width, Conc2.outChannel, 1, 3);

	workspaceSize = MAXGRID * sizeof(convtype);
	workspaceSize = (workspaceSize > C1.workspaceSize) ? workspaceSize : C1.workspaceSize;
	workspaceSize = (workspaceSize > C2_1.workspaceSize) ? workspaceSize : C2_1.workspaceSize;
	workspaceSize = (workspaceSize > C2_2.workspaceSize) ? workspaceSize : C2_2.workspaceSize;
	workspaceSize = (workspaceSize > C3_1.workspaceSize) ? workspaceSize : C3_1.workspaceSize;
	workspaceSize = (workspaceSize > C3_2.workspaceSize) ? workspaceSize : C3_2.workspaceSize;
	workspaceSize = (workspaceSize > C4.workspaceSize) ? workspaceSize : C4.workspaceSize;
	cudaDeviceSynchronize();
	check(cudaMalloc(&workspace, workspaceSize));
}
int qvrcnn::load_para(char *filename)
{
	FILE *fp;
	if (fopen_s(&fp, filename, "rb"))
	{
		printf("cannot open model file.\n");
		exit(1);
	}
	C1.load_para(fp);
	C2_1.load_para(fp);
	C2_2.load_para(fp);
	C3_1.load_para(fp);
	C3_2.load_para(fp);
	C4.load_para(fp);
	fclose(fp);
	return 0;
}
int qvrcnn::load_static_para(char *filename)
{
	FILE *fp;
	if (fopen_s(&fp, filename, "rb"))
	{
		printf("cannot open model file.\n");
		exit(1);
	}
	C1.load_static_para(fp);
	C2_1.load_static_para(fp);
	C2_2.load_static_para(fp);
	C3_1.load_static_para(fp);
	C3_2.load_static_para(fp);
	C4.load_static_para(fp);
	fclose(fp);
	return 0;
}
int qvrcnn::load_data(datatype *input)
{
	I1.load(input);
	return 0;
}
#if defined(INT8x4_EXT_CONFIG)||defined(INT8_EXT_CONFIG)
int save_steps(int *max_u, const char*filename)
{
	FILE*fp;
	if (fopen_s(&fp, filename, "ab"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}
	fwrite(max_u, sizeof(int), 1, fp);
	fclose(fp);
	return 0;
}
int qvrcnn::forward(void)
{
	int layer=0;

	//input layer
	I1.ppro();

	//layer 1
	layer = 1;
	adjustBasic<<<1,C1.outChannel>>>(steps, (btype*)C1.b, (btype*)C1.b_adj, layer-1);
	C1.ConvForward(cudnnHandle, I1.xDesc, I1.x_ppro, workspace, workspaceSize);
#ifdef MEM_DBG
	C1.viewmem((xwtype*)I1.x_ppro);
#endif
	//C1.activate(cudnnHandle);
	cudaDeviceSynchronize();
	//C1.quantize_out(workspace);
	//C1.quantize_out_fix(2689);//ori=53788
	C1.quantize_out_static();
#ifdef MEM_DBG
	C1.viewmem((xwtype*)I1.x_ppro);
#endif
	insert_w(C1.step_w, layer);
	insert_y(C1.step_y, layer);

	//layer 2
	layer = 2;
	adjustBasic <<<1, C2_1.outChannel >>>(steps, (btype*)C2_1.b, (btype*)C2_1.b_adj, layer-1);
	adjustBasic <<<1, C2_2.outChannel >>>(steps, (btype*)C2_2.b, (btype*)C2_2.b_adj, layer - 1);
	C2_1.ConvForward(cudnnHandle, C1.vDesc, C1.v, workspace, workspaceSize);
#ifdef MEM_DBG
	C2_1.viewmem((xwtype*)C1.v);
#endif
	C2_1.activate(cudnnHandle);
	C2_2.ConvForward(cudnnHandle, C1.vDesc, C1.v, workspace, workspaceSize);
#ifdef MEM_DBG
	C2_2.viewmem((xwtype*)C1.v);
#endif
	C2_2.activate(cudnnHandle);
	//Conc1.concat(&C2_1, &C2_2, workspace);
	Conc1.concat_blu(&C2_1, &C2_2);
	insert_w(C2_1.step_w, layer);
	insert_y(C2_1.step_y, layer);

	//layer 3
	layer = 3;
	adjustBasic <<<1, C3_1.outChannel>>>(steps, (btype*)C3_1.b, (btype*)C3_1.b_adj, layer-1);
	adjustBasic <<<1, C3_2.outChannel>>>(steps, (btype*)C3_2.b, (btype*)C3_2.b_adj, layer-1);
	C3_1.ConvForward(cudnnHandle, Conc1.concDesc, Conc1.conc, workspace, workspaceSize);
#ifdef MEM_DBG
	C3_1.viewmem((xwtype*)Conc1.conc);
#endif
	C3_1.activate(cudnnHandle);
	C3_2.ConvForward(cudnnHandle, Conc1.concDesc, Conc1.conc, workspace, workspaceSize);
#ifdef MEM_DBG
	C3_2.viewmem((xwtype*)Conc1.conc);
#endif
	C3_2.activate(cudnnHandle);
	//Conc2.concat(&C3_1, &C3_2, workspace);
	Conc2.concat_blu(&C3_1, &C3_2);
	insert_w(C3_1.step_w, layer);
	insert_y(C3_1.step_y, layer);

	//layer 4
	layer = 4;
	adjustBasic <<<1, C4.outChannel>>>(steps, (btype*)C4.b, (btype*)C4.b_adj, layer-1);
	C4.ConvForward(cudnnHandle, Conc2.concDesc, Conc2.conc, workspace, workspaceSize);
#ifdef MEM_DBG
	C4.viewmem((xwtype*)Conc2.conc);
#endif
	insert_w(C4.step_w, layer);
	
	//restore
	cudaDeviceSynchronize();
	//adjustOutput<<<GRIDSIZE, BLOCKSIZE>>>(steps, (convtype*)C4.u, (xwtype*)C4.v, layer,C4.uSize,GRIDSIZE*BLOCKSIZE);//scale n times
	adjustOutput_static<<<GRIDSIZE, BLOCKSIZE>>>((convtype*)C4.u, (xwtype*)C4.v, 141, 16, C4.uSize, GRIDSIZE*BLOCKSIZE);
#ifdef MEM_DBG
	C4.viewmem((xwtype*)C4.v);
#endif
	cudaDeviceSynchronize();
	I1.applyRes((xwtype*)C4.v);
	save_steps(&C1.max_u, "max_u_C1.data");
	//save_steps(&C3_2.max_u, "max_u_C3_2.data");
	//save_b_adj("b_adj.data");
	return 0;
}
int qvrcnn::forward_blu(void)
{
	int layer = 0;

	//input layer
	I1.ppro();

	//layer 1
	//layer = 1;
	//adjustBasic <<<1, C1.outChannel >>>(steps, (btype*)C1.b, (btype*)C1.b_adj, layer - 1);
	C1.ConvForward_static(cudnnHandle, I1.xDesc, I1.x_ppro, workspace, workspaceSize);
#ifdef MEM_DBG
	C1.viewmem((xwtype*)I1.x_ppro);
#endif
	cudaDeviceSynchronize();
	C1.quantize_out_blu();
#ifdef MEM_DBG
	C1.viewmem((xwtype*)I1.x_ppro);
#endif
	//insert_w(C1.step_w, layer);
	//insert_y(C1.step_y, layer);

	//layer 2
	layer = 2;
	//adjustBasic << <1, C2_1.outChannel >> >(steps, (btype*)C2_1.b, (btype*)C2_1.b_adj, layer - 1);
	//adjustBasic << <1, C2_2.outChannel >> >(steps, (btype*)C2_2.b, (btype*)C2_2.b_adj, layer - 1);
	C2_1.ConvForward_static(cudnnHandle, C1.vDesc, C1.v, workspace, workspaceSize);
#ifdef MEM_DBG
	C2_1.viewmem((xwtype*)C1.v);
#endif
	C2_2.ConvForward_static(cudnnHandle, C1.vDesc, C1.v, workspace, workspaceSize);
#ifdef MEM_DBG
	C2_2.viewmem((xwtype*)C1.v);
#endif
	Conc1.concat_blu(&C2_1, &C2_2);
	//insert_w(C2_1.step_w, layer);
	//insert_y(C2_1.step_y, layer);

	//layer 3
	layer = 3;
	//adjustBasic << <1, C3_1.outChannel >> >(steps, (btype*)C3_1.b, (btype*)C3_1.b_adj, layer - 1);
	//adjustBasic << <1, C3_2.outChannel >> >(steps, (btype*)C3_2.b, (btype*)C3_2.b_adj, layer - 1);
	C3_1.ConvForward_static(cudnnHandle, Conc1.concDesc, Conc1.conc, workspace, workspaceSize);
#ifdef MEM_DBG
	C3_1.viewmem((xwtype*)Conc1.conc);
#endif
	C3_2.ConvForward_static(cudnnHandle, Conc1.concDesc, Conc1.conc, workspace, workspaceSize);
#ifdef MEM_DBG
	C3_2.viewmem((xwtype*)Conc1.conc);
#endif
	Conc2.concat_blu(&C3_1, &C3_2);
	//insert_w(C3_1.step_w, layer);
	//insert_y(C3_1.step_y, layer);

	//layer 4
	layer = 4;
	//adjustBasic << <1, C4.outChannel >> >(steps, (btype*)C4.b, (btype*)C4.b_adj, layer - 1);
	C4.ConvForward_static(cudnnHandle, Conc2.concDesc, Conc2.conc, workspace, workspaceSize);
#ifdef MEM_DBG
	C4.viewmem((xwtype*)Conc2.conc);
#endif
#ifdef MEM_DBG
	//restore
	C4.quantize_out_static();
	C4.viewmem((xwtype*)Conc2.conc);
#endif
	I1.applyRes_y((convtype*)C4.u, C4.mul, C4.shift);
#ifdef MEM_DBG
	I1.viewmem((xwtype*)C4.v);
#endif
	//save_steps(&C1.max_u, "max_u_C1.data");
	//save_steps(&C3_2.max_u, "max_u_C3_2.data");
	//save_b_adj("b_adj.data");
	return 0;
}
#elif defined(FLOAT_CONFIG)
int qvrcnn::forward(void)
{
	int layer = 0;

	//input layer
	I1.ppro();

	//layer 1
	layer = 1;
	C1.ConvForward(cudnnHandle, I1.xDesc, I1.x_ppro, workspace, workspaceSize);
	//C1.viewmem((xwtype*)I1.x_ppro);
	C1.activate(cudnnHandle);

	//layer 2
	layer = 2;
	C2_1.ConvForward(cudnnHandle, C1.uDesc, C1.u, workspace, workspaceSize);
	//C2_1.viewmem((xwtype*)C1.u);
	C2_1.activate(cudnnHandle);
	C2_2.ConvForward(cudnnHandle, C1.uDesc, C1.u, workspace, workspaceSize);
	//C2_2.viewmem((xwtype*)C1.u);
	C2_2.activate(cudnnHandle);
	Conc1.concat(&C2_1, &C2_2, workspace);

	//layer 3
	layer = 3;
	C3_1.ConvForward(cudnnHandle, Conc1.concDesc, Conc1.conc, workspace, workspaceSize);
	//C3_1.viewmem((xwtype*)Conc1.conc);
	C3_1.activate(cudnnHandle);
	C3_2.ConvForward(cudnnHandle, Conc1.concDesc, Conc1.conc, workspace, workspaceSize);
	//C3_2.viewmem((xwtype*)Conc1.conc);
	C3_2.activate(cudnnHandle);
	Conc2.concat(&C3_1, &C3_2, workspace);

	//layer 4
	layer = 4;
	C4.ConvForward(cudnnHandle, Conc2.concDesc, Conc2.conc, workspace, workspaceSize);
	//C4.viewmem((xwtype*)Conc2.conc);																		  //C4.viewmem();
	cudaDeviceSynchronize();
	I1.applyRes((xwtype*)C4.u);
	//I1.viewmem((xwtype*)C4.u);
	return 0;
}
#endif

int qvrcnn::save_b_adj(const char*filename)
{
	FILE*fp;
	if (fopen_s(&fp, filename, "a+"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}
	fwrite(C1.b_adj, sizeof(btype), C1.outChannel, fp);
	fwrite(C2_1.b_adj, sizeof(btype), C2_1.outChannel, fp);
	fwrite(C2_2.b_adj, sizeof(btype), C2_2.outChannel, fp);
	fwrite(C3_1.b_adj, sizeof(btype), C3_1.outChannel, fp);
	fwrite(C3_2.b_adj, sizeof(btype), C3_2.outChannel, fp);
	fwrite(C4.b_adj, sizeof(btype), C4.outChannel, fp);
	fclose(fp);
	return 0;
}
void qvrcnn::insert_w(int stepw, int layer)//layer means current layer
{
	int i, j;
	for (i = 0;i < layer - 1;i++)
		if (steps.stepw[i] < stepw)
		{
			for (j = layer - 1;j > i;j--)steps.stepw[j] = steps.stepw[j - 1];
			steps.stepw[j] = stepw;
			return;
		}
	steps.stepw[i] = stepw;
	return;
}
void qvrcnn::insert_y(int stepy, int layer)//layer means current layer
{
	int i, j;
	for (i = 0;i < layer - 1;i++)
		if (steps.stepy[i] > stepy)
		{
			for (j = layer - 1;j > i;j--)steps.stepy[j] = steps.stepy[j - 1];
			steps.stepy[j] = stepy;
			return;
		}
	steps.stepy[i] = stepy;
	return;
}
qvrcnn::~qvrcnn()
{
	cudnnDestroy(cudnnHandle);
	check(cudaFree(workspace));
}
__global__ void adjustBasic(step_parameters steps, btype*b, btype *b_adj, int n)//scale n times
{
	long long temp = b[threadIdx.x];
	int w_num, y_num;
	for (w_num = 0;w_num < n;w_num++)temp *= steps.stepw[w_num];
	for (y_num = 0;y_num < n;y_num++)
	{
		if (temp > 0)
			temp = (temp + (steps.stepy[y_num] >> 1)) / steps.stepy[y_num];
		else
			temp = (temp - (steps.stepy[y_num] >> 1)) / steps.stepy[y_num];
	}
	b_adj[threadIdx.x] = temp;
}
/*{
	long long temp = b[threadIdx.x];
	int w_num, y_num;
	for (y_num = w_num = 0;y_num < n;y_num++)
	{
		while (w_num < n && (temp*steps.stepw[w_num]) / steps.stepw[w_num] == temp)
		{
			temp *= steps.stepw[w_num];
			w_num++;
		}
		if (temp > 0)
			temp = (temp + steps.stepy[y_num] >> 1) / steps.stepy[y_num];// + (temp%steps.stepy[y_num] + steps.stepy[y_num] / 2) / steps.stepy[y_num];//avoid overflow
		else
			temp = (temp - steps.stepy[y_num] >> 1) / steps.stepy[y_num];// + (temp%steps.stepy[y_num] - steps.stepy[y_num] / 2) / steps.stepy[y_num];//avoid overflow
	}
	b_adj[threadIdx.x] = temp;
}*/
//scale n times,for num elements in total and gridSize threads in total
__global__ void adjustOutput(step_parameters steps, convtype*o, xwtype *o_adj, int n, int num, int gridSize)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int w_num, y_num;
	while (tid < num)
	{
		long long temp = o[tid];
		for (y_num = 0;y_num < n - 1;y_num++)temp *= steps.stepy[y_num];
		if (temp > 0)
			for (w_num = n - 1;w_num >= 0;w_num--)
				temp = (temp + (steps.stepw[w_num] >> 1)) / steps.stepw[w_num];
		else
			for (w_num = n - 1;w_num >= 0;w_num--)
				temp = (temp - (steps.stepw[w_num] >> 1)) / steps.stepw[w_num];
		o_adj[tid] = temp;
		tid += gridSize;
	}
}
__global__ void adjustOutput_static(convtype*o, xwtype *o_adj, int multiplier, int shifts, int num, int gridSize)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int temp;
	int bias = 1 << shifts - 1;
	while (tid < num)
	{
		temp = o[tid];
		o_adj[tid] = temp*multiplier + bias >> shifts;
		tid += gridSize;
	}
}
int layer_HWCN2NCHW_VECT_C(FILE* fp_in, FILE* fp_out, int ksize, int inChannel, int outChannel)
{
	int wSize = ksize*ksize*inChannel*outChannel, wSize_out;
	char *w_in = new char[wSize];
	char *w_out;
	int *b = new int[outChannel];
	fread(b, sizeof(int), 1, fp_in);
	fwrite(b, sizeof(int), 1, fp_out);
	fread(w_in, sizeof(char), wSize, fp_in);
	w_out = HWCN2NCHW_VECT_C_CPU(w_in, ksize, ksize, inChannel, outChannel, &wSize_out);
	fwrite(w_out, sizeof(char), wSize_out, fp_out);
	fread(b, sizeof(int), outChannel, fp_in);
	fwrite(b, sizeof(int), outChannel, fp_out);
	free(w_in);
	free(w_out);
	free(b);
	return 0;
}
int model_HWCN2NCHW_VECT_C(const char* filein, const char *fileout, int qp)
{
	FILE *fp_in, *fp_out;
	char filename[100];
	sprintf_s(filename, filein, qp);
	if (fopen_s(&fp_in, filename, "rb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}
	sprintf_s(filename, fileout, qp);
	if (fopen_s(&fp_out, filename, "wb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}

	layer_HWCN2NCHW_VECT_C(fp_in, fp_out, 5, 1, 64);
	layer_HWCN2NCHW_VECT_C(fp_in, fp_out, 3, 64, 32);
	layer_HWCN2NCHW_VECT_C(fp_in, fp_out, 5, 64, 16);
	layer_HWCN2NCHW_VECT_C(fp_in, fp_out, 3, 48, 16);
	layer_HWCN2NCHW_VECT_C(fp_in, fp_out, 1, 48, 32);
	layer_HWCN2NCHW_VECT_C(fp_in, fp_out, 3, 48, 1);

	fclose(fp_in);
	fclose(fp_out);
	return 0;
}
int layer_HWCN2NCHW(FILE* fp_in, FILE* fp_out, int ksize, int inChannel, int outChannel)
{
	int wSize = ksize*ksize*inChannel*outChannel, wSize_out;
	xwtype *w_in = new xwtype[wSize];
	xwtype *w_out;
	btype *b = new btype[outChannel];
#if defined(INT8x4_EXT_CONFIG)||defined(INT8_EXT_CONFIG)
	fread(b, sizeof(btype), 1, fp_in);
	fwrite(b, sizeof(btype), 1, fp_out);
#endif
	fread(w_in, sizeof(xwtype), wSize, fp_in);
	w_out = HWCN2NCHW_CPU(w_in, ksize, ksize, inChannel, outChannel, &wSize_out);
	fwrite(w_out, sizeof(xwtype), wSize_out, fp_out);
	fread(b, sizeof(btype), outChannel, fp_in);
	fwrite(b, sizeof(btype), outChannel, fp_out);
	free(w_in);
	free(w_out);
	free(b);
	return 0;
}
int model_HWCN2NCHW(const char* filein, const char *fileout, int qp)
{
	FILE *fp_in, *fp_out;
	char filename[100];
	sprintf_s(filename, filein, qp);
	if (fopen_s(&fp_in, filename, "rb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}
	sprintf_s(filename, fileout, qp);
	if (fopen_s(&fp_out, filename, "wb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}

	layer_HWCN2NCHW(fp_in, fp_out, 5, 1, 64);
	layer_HWCN2NCHW(fp_in, fp_out, 3, 64, 32);
	layer_HWCN2NCHW(fp_in, fp_out, 5, 64, 16);
	layer_HWCN2NCHW(fp_in, fp_out, 3, 48, 16);
	layer_HWCN2NCHW(fp_in, fp_out, 1, 48, 32);
	layer_HWCN2NCHW(fp_in, fp_out, 3, 48, 1);

	fclose(fp_in);
	fclose(fp_out);
	return 0;
}
int layer_NCWH2HWCN(FILE* fp_in, FILE* fp_out, int ksize, int inChannel, int outChannel)
{
	int wSize = ksize*ksize*inChannel*outChannel, wSize_out;
	char *w_in = new char[wSize];
	char *w_out;
	int *b = new int[outChannel];
	fread(b, sizeof(int), 1, fp_in);
	fwrite(b, sizeof(int), 1, fp_out);
	fread(w_in, sizeof(char), wSize, fp_in);
	w_out = NCWH2HWCN_CPU(w_in, ksize, ksize, inChannel, outChannel, &wSize_out);
	fwrite(w_out, sizeof(char), wSize_out, fp_out);
	fread(b, sizeof(int), outChannel, fp_in);
	fwrite(b, sizeof(int), outChannel, fp_out);
	return 0;
}
int model_NCWH2HWCN(const char* filein, const char *fileout, int qp)
{
	FILE *fp_in, *fp_out;
	char filename[100];
	sprintf_s(filename, filein, qp);
	if (fopen_s(&fp_in, filename, "rb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}
	sprintf_s(filename, fileout, qp);
	if (fopen_s(&fp_out, filename, "wb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}

	layer_NCWH2HWCN(fp_in, fp_out, 5, 1, 64);
	layer_NCWH2HWCN(fp_in, fp_out, 3, 64, 32);
	layer_NCWH2HWCN(fp_in, fp_out, 5, 64, 16);
	layer_NCWH2HWCN(fp_in, fp_out, 3, 48, 16);
	layer_NCWH2HWCN(fp_in, fp_out, 1, 48, 32);
	layer_NCWH2HWCN(fp_in, fp_out, 3, 48, 1);

	fclose(fp_in);
	fclose(fp_out);
	return 0;
}
int layer_qfp_HWCN2NCHW_VECT_C(FILE* fp_in, FILE* fp_out, int ksize, int inChannel, int outChannel)
{
	int wSize = ksize*ksize*inChannel*outChannel, wSize_out;
	char *w_in = new char[wSize];
	char *w_out;
	int *b = new int[outChannel];

	fread(w_in, sizeof(char), wSize, fp_in);
	fread(b, sizeof(int), outChannel, fp_in);
	
	w_out = HWCN2NCHW_VECT_C_CPU(w_in, ksize, ksize, inChannel, outChannel, &wSize_out);
	fwrite(w_out, sizeof(char), wSize_out, fp_out);
	fwrite(b, sizeof(int), outChannel, fp_out);
	
	//blu_q,mul,shift
	fread(b, sizeof(int), 1, fp_in);
	fwrite(b, sizeof(int), 1, fp_out);
	fread(b, sizeof(int), 1, fp_in);
	fwrite(b, sizeof(int), 1, fp_out);
	fread(b, sizeof(int), 1, fp_in);
	fwrite(b, sizeof(int), 1, fp_out);
	return 0;
}
int model_qfp_HWCN2NCHW_VECT_C(const char* filein, const char *fileout, int qp)
{
	FILE *fp_in, *fp_out;
	char filename[100];
	sprintf_s(filename, filein, qp);
	if (fopen_s(&fp_in, filename, "rb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}
	sprintf_s(filename, fileout, qp);
	if (fopen_s(&fp_out, filename, "wb"))
	{
		printf("failed to open file %s", filename);
		exit(1);
	}

	layer_qfp_HWCN2NCHW_VECT_C(fp_in, fp_out, 5, 1, 64);
	layer_qfp_HWCN2NCHW_VECT_C(fp_in, fp_out, 3, 64, 32);
	layer_qfp_HWCN2NCHW_VECT_C(fp_in, fp_out, 5, 64, 16);
	layer_qfp_HWCN2NCHW_VECT_C(fp_in, fp_out, 3, 48, 16);
	layer_qfp_HWCN2NCHW_VECT_C(fp_in, fp_out, 1, 48, 32);
	layer_qfp_HWCN2NCHW_VECT_C(fp_in, fp_out, 3, 48, 1);

	fclose(fp_in);
	fclose(fp_out);
	return 0;
}
