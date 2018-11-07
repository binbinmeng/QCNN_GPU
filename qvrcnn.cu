#include "qvrcnn.cuh"
#include <stdio.h>

qvrcnn::qvrcnn(int gpu_num, int batch, int channel, int height, int width)//build qvrcnn
{
	check(cudaSetDevice(gpu_num));
	check(cudnnCreate(&cudnnHandle));
	this->batch = batch, this->channel = channel, this->height = height, this->width = width;
	
	I1.build(batch, channel, height, width);
	C1.build(cudnnHandle, I1.xDesc, batch, height, width, I1.outChannel, 64, 5);
	C2_1.build(cudnnHandle, C1.vDesc, batch, height, width, 64, 32, 3);
	C2_2.build(cudnnHandle, C1.vDesc, batch, height, width, 64, 16, 5);
	Conc1.build(batch, height, width, 32, 16);
	C3_1.build(cudnnHandle, Conc1.concDesc, batch, height, width, 48, 16, 3);
	C3_2.build(cudnnHandle, Conc1.concDesc, batch, height, width, 48, 32, 1);
	Conc2.build(batch, height, width, 16, 32);
	C4.build(cudnnHandle, Conc2.concDesc, batch, height, width, 48, 1, 5);

	workspaceSize = GRIDSIZE * sizeof(convtype);
	workspaceSize = (workspaceSize > C1.workspaceSize) ? workspaceSize : C1.workspaceSize;
	workspaceSize = (workspaceSize > C2_1.workspaceSize) ? workspaceSize : C2_1.workspaceSize;
	workspaceSize = (workspaceSize > C2_2.workspaceSize) ? workspaceSize : C2_2.workspaceSize;
	workspaceSize = (workspaceSize > C3_1.workspaceSize) ? workspaceSize : C3_1.workspaceSize;
	workspaceSize = (workspaceSize > C3_2.workspaceSize) ? workspaceSize : C3_2.workspaceSize;
	workspaceSize = (workspaceSize > C4.workspaceSize) ? workspaceSize : C4.workspaceSize;
	cudaMalloc(&workspace, workspaceSize);
}
int qvrcnn::load_para(char *filename)
{
	FILE *fp;
	if (fopen_s(&fp, filename, "rb"))
	{
		printf("cannot open data file.\n");
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
int qvrcnn::load_data(datatype *input)
{
	I1.load(input);
	return 0;
}
int qvrcnn::forward(void)
{
	int layer=0;

	//input layer
	I1.ppro();

	//layer 1
	layer = 1;
	adjustBasic<<<1,64>>>(steps, (btype*)C1.b, (btype*)C1.b_adj, layer-1);
	C1.ConvForward(cudnnHandle, I1.xDesc, I1.x_ppro, workspace, workspaceSize);
	C1.activate(cudnnHandle);
	C1.quantize_out(workspace);
	//C1.viewmem((xwtype*)I1.x_ppro);
	insert_w(C1.step_w, layer);
	insert_y(C1.step_y, layer);

	//layer 2
	layer = 2;
	adjustBasic <<<1, 32>>>(steps, (btype*)C2_1.b, (btype*)C2_1.b_adj, layer-1);
	//C2_1.viewmem();
	adjustBasic <<<1, 16>>>(steps, (btype*)C2_2.b, (btype*)C2_2.b_adj, layer-1);
	C2_1.ConvForward(cudnnHandle, C1.vDesc, C1.v, workspace, workspaceSize);
	C2_1.activate(cudnnHandle);
	//C2_1.viewmem();
	C2_2.ConvForward(cudnnHandle, C1.vDesc, C1.v, workspace, workspaceSize);
	C2_2.activate(cudnnHandle);
	//C2_2.viewmem();
	Conc1.concat(&C2_1, &C2_2, workspace);
	insert_w(C2_1.step_w, layer);
	insert_y(C2_1.step_y, layer);

	//layer 3
	layer = 3;
	adjustBasic <<<1, 16>>>(steps, (btype*)C3_1.b, (btype*)C3_1.b_adj, layer-1);
	adjustBasic <<<1, 32>>>(steps, (btype*)C3_2.b, (btype*)C3_2.b_adj, layer-1);
	C3_1.ConvForward(cudnnHandle, Conc1.concDesc, Conc1.conc, workspace, workspaceSize);
	C3_1.activate(cudnnHandle);
	//C3_1.viewmem();
	C3_2.ConvForward(cudnnHandle, Conc1.concDesc, Conc1.conc, workspace, workspaceSize);
	C3_2.activate(cudnnHandle);
	//C3_2.viewmem();
	Conc2.concat(&C3_1, &C3_2, workspace);
	insert_w(C3_1.step_w, layer);
	insert_y(C3_1.step_y, layer);

	//layer 4
	layer = 4;
	adjustBasic << <1, 1 >> >(steps, (btype*)C4.b, (btype*)C4.b_adj, layer-1);
	C4.ConvForward(cudnnHandle, Conc2.concDesc, Conc2.conc, workspace, workspaceSize);
	//C4.viewmem();
	insert_w(C4.step_w, layer);
	
	//restore
	cudaDeviceSynchronize();
	adjustOutput<<<HEIGHT, WIDTH>>>(steps, (convtype*)C4.u, (xwtype*)C4.v, layer);//scale n times
	//C4.viewmem();
	cudaDeviceSynchronize();
	I1.applyRes((xwtype*)C4.v);
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
__global__ void adjustOutput(step_parameters steps, convtype*o, xwtype *o_adj, int n)//scale n times
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int w_num, y_num;
	long long temp = o[tid];
	for (y_num = 0;y_num < n - 1;y_num++)temp *= steps.stepy[y_num];
	if (temp > 0)
		for (w_num = n - 1;w_num >= 0;w_num--)
			temp = (temp + (steps.stepw[w_num] >> 1)) / steps.stepw[w_num];
	else
		for (w_num = n - 1;w_num >= 0;w_num--)
			temp = (temp - (steps.stepw[w_num] >> 1)) / steps.stepw[w_num];
	o_adj[tid] = temp;
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
	layer_HWCN2NCHW_VECT_C(fp_in, fp_out, 5, 48, 1);

	fclose(fp_in);
	fclose(fp_out);
	return 0;
}
/*
//init vrcnn
void vrcnnsetup(VRCNN* vrcnn, int inputwidth, int inputheight)
{
	vrcnn->layerNum = 4;

	vrcnn->C1 = initTFLayer(inputwidth, inputheight, 5, 1, 64);
	vrcnn->C2_1 = initTFLayer(inputwidth, inputheight, 3, 64, 32);
	vrcnn->C2_2 = initTFLayer(inputwidth, inputheight, 5, 64, 16);
	vrcnn->C3_1 = initTFLayer(inputwidth, inputheight, 3, 48, 16);
	vrcnn->C3_2 = initTFLayer(inputwidth, inputheight, 1, 48, 32);
	vrcnn->C4 = initTFLayer(inputwidth, inputheight, 5, 48, 1);
}
// 导入vrcnn的数据
void importvrcnn(VRCNN* vrcnn, const char* filename)
{
	FILE  *fp = NULL;
	if (fopen_s(&fp, filename, "rb"))
		printf("Cannot open weight file.\n");
	importlayer(vrcnn->C1, fp);
	importlayer(vrcnn->C2_1, fp);
	importlayer(vrcnn->C2_2, fp);
	importlayer(vrcnn->C3_1, fp);
	importlayer(vrcnn->C3_2, fp);
	importlayer(vrcnn->C4, fp);
	fclose(fp);
}

void inspectValue(QCovLayer* L)
{
	int i, j, k, l;
	for (i = 0; i < L->mapSize; i++)
		for (j = 0; j < L->mapSize; j++)
			for (k = 0; k < L->inChannels; k++)
				for (l = 0; l < L->outChannels; l++)
					if (L->mapData[i][j][k][l] > QTHRESHOLD)
						printf("\nWarning: quantized value overflow!\n");
}

Q_VRCNN* quantizeVRCNN(VRCNN* vrcnn)
{
	Q_VRCNN* qvrcnn = (Q_VRCNN*)malloc(sizeof(Q_VRCNN));
	qvrcnn->layerNum = 4;
	qvrcnn->C1 = quantizeLayer(vrcnn->C1);
	qvrcnn->C2_1 = quantizeLayer(vrcnn->C2_1);
	qvrcnn->C2_2 = quantizeLayer(vrcnn->C2_2);
	qvrcnn->C3_1 = quantizeLayer(vrcnn->C3_1);
	qvrcnn->C3_2 = quantizeLayer(vrcnn->C3_2);
	qvrcnn->C4 = quantizeLayer(vrcnn->C4);
	inspectValue(qvrcnn->C1);
	inspectValue(qvrcnn->C2_1);
	inspectValue(qvrcnn->C2_2);
	inspectValue(qvrcnn->C3_1);
	inspectValue(qvrcnn->C3_2);
	inspectValue(qvrcnn->C4);
	return qvrcnn;
}

void saveQ_VRCNN(Q_VRCNN* qvrcnn, const char* name)
{
	FILE* fp;
	if (fopen_s(&fp, name, "wb"))
	{
		printf("failed to open file %s", name);
		exit(1);
	}
	saveQLayer(qvrcnn->C1, fp);
	saveQLayer(qvrcnn->C2_1, fp);
	saveQLayer(qvrcnn->C2_2, fp);
	saveQLayer(qvrcnn->C3_1, fp);
	saveQLayer(qvrcnn->C3_2, fp);
	saveQLayer(qvrcnn->C4, fp);
	fclose(fp);
}

void readQ_VRCNN(Q_VRCNN* qvrcnn, const char* name)
{
	FILE* fp;
	if (fopen_s(&fp, name, "rb"))
	{
		printf("failed to open file %s", name);
		exit(1);
	}
	readQLayer(qvrcnn->C1, fp);
	readQLayer(qvrcnn->C2_1, fp);
	readQLayer(qvrcnn->C2_2, fp);
	readQLayer(qvrcnn->C3_1, fp);
	readQLayer(qvrcnn->C3_2, fp);
	readQLayer(qvrcnn->C4, fp);
	fclose(fp);
}
*/