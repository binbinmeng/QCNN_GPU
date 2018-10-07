#include "qvrcnn.cuh"

qvrcnn::qvrcnn(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int batch, int channel, int inheight, int inwidth)//build qvrcnn
{
	height = inheight, width = inwidth;
	C1.build(cudnnHandle, xDesc, channel, 64, height, width, 5);
	C2_1.build(cudnnHandle, C1.yDesc, 64, 32, height, width, 3);
	C2_2.build(cudnnHandle, C1.yDesc, 64, 16, height, width, 5);
	setConcat(conc1_Desc, conc1, 48);
	C3_1.build(cudnnHandle, conc1_Desc, 48, 16, height, width, 3);
	C3_2.build(cudnnHandle, conc1_Desc, 48, 32, height, width, 1);
	setConcat(conc2_Desc, conc2, 48);
	C4.build(cudnnHandle, conc2_Desc, 48, 1, height, width, 5);
	workspaceSize = GRIDSIZE * sizeof(cudnnType);
	workspaceSize = (workspaceSize > C1.buffer_size()) ? workspaceSize : C1.buffer_size();
	workspaceSize = (workspaceSize > C2_1.buffer_size()) ? workspaceSize : C2_1.buffer_size();
	workspaceSize = (workspaceSize > C2_2.buffer_size()) ? workspaceSize : C2_2.buffer_size();
	workspaceSize = (workspaceSize > C3_1.buffer_size()) ? workspaceSize : C3_1.buffer_size();
	workspaceSize = (workspaceSize > C3_2.buffer_size()) ? workspaceSize : C3_2.buffer_size();
	workspaceSize = (workspaceSize > C4.buffer_size()) ? workspaceSize : C4.buffer_size();
	cudaMalloc(&workspace, workspaceSize);
}
int qvrcnn::setConcat(cudnnTensorDescriptor_t conc_Desc, void *conc, int channel)
{
	cudnnCreateTensorDescriptor(&conc_Desc);
	check(cudnnSetTensor4dDescriptor(conc_Desc,
		CUDNN_TENSOR_NCHW,
		CUDNNTYPE,
		1, channel, height, width));
	check(cudaMalloc(&conc, sizeof(cudnnType)*channel*height*width));
}
int qvrcnn::load(const char *filename)
{
	FILE *fp;
	if (fopen_s(&fp, filename, "rb") == NULL)
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
int qvrcnn::forward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void *x)
{
	int layer;

	//layer 1
	layer = 1;
	insert_w(C1.step_w, layer);
	adjustBasic<<<1,64>>>(steps, (cudnnType*)C1.b, (cudnnType*)C1.b, layer);
	C1.ConvForward(cudnnHandle, xDesc, x, workspace, workspaceSize);
	C1.quantize_out(workspace);
	insert_y(C1.step_y, layer);

	//layer 2
	layer = 2;
	insert_w(C2_1.step_w, layer);
	adjustBasic << <1, 32 >> >(steps, (cudnnType*)C2_1.b, (cudnnType*)C2_1.b, layer);
	adjustBasic << <1, 32 >> >(steps, (cudnnType*)C2_2.b, (cudnnType*)C2_2.b, layer);
	C2_1.ConvForward(cudnnHandle, C1.yDesc, C1.y, workspace, workspaceSize);
	C2_2.ConvForward(cudnnHandle, C1.yDesc, C1.y, workspace, workspaceSize);
	concat(C2_1, C2_2, conc1);
	insert_y(C2_1.step_y, layer);

	//layer 3
	layer = 3;
	insert_w(C3_1.step_w, layer);
	adjustBasic << <1, 32 >> >(steps, (cudnnType*)C3_1.b, (cudnnType*)C3_1.b, layer);
	adjustBasic << <1, 32 >> >(steps, (cudnnType*)C3_2.b, (cudnnType*)C3_2.b, layer);
	C3_1.ConvForward(cudnnHandle, conc1_Desc, conc1, workspace, workspaceSize);
	C3_2.ConvForward(cudnnHandle, conc1_Desc, conc1, workspace, workspaceSize);
	concat(C3_1, C3_2, conc2);
	insert_y(C3_1.step_y, layer);

	//layer 4
	layer = 4;
	insert_w(C4.step_w, layer);
	adjustBasic << <1, 64 >> >(steps, (cudnnType*)C4.b, (cudnnType*)C4.b, layer);
	C4.ConvForward(cudnnHandle, conc2_Desc, conc2, workspace, workspaceSize);
	
	//restore
	adjustOutput<<<GRIDSIZE,BLOCKSIZE>>>(steps, (cudnnType*)C4.u, (cudnnType*)C4.y, layer);//scale n times
	return 0;
}
int qvrcnn::concat(CovLayer C1, CovLayer C2, void *p)
{
	int i, j, k;
	cudnnType max1, max2;
	int stepy1, stepy2;
	findMax((cudnnType*)C1.u, (cudnnType*)workspace, C1.paras.outSize, &max1);
	findMax((cudnnType*)C2.u, (cudnnType*)workspace, C2.paras.outSize, &max1);
	if (max1 > THRESHOLD)
		stepy1 = max1 / (THRESHOLD + 1) + 1;
	else
		stepy1 = 1;
	if (max2 > THRESHOLD)
		stepy2 = max2 / (THRESHOLD + 1) + 1;
	else
		stepy2 = 1;
	//对stepy进行调整以保持ratio一致,结果不能太小
	if (C1.step_w*stepy2 > C2.step_w*stepy1)
		stepy1 = C1.step_w*stepy2 / C2.step_w;
	else
		stepy2 = C2.step_w*stepy1 / C1.step_w;
	C1.step_y = stepy1;
	C1.step_y = stepy2;

	//free C1.y and C2.y, compute concat
	cudaFree(C1.y);
	cudaFree(C2.y);
	if (stepy1 > 1)
		VectorDiv << <GRIDSIZE, BLOCKSIZE >> > ((cudnnType*)C1.u, (cudnnType*)p, stepy1, C1.paras.outSize);
	else
		cudaMemcpy(p, C1.u, C1.paras.outSize * sizeof(cudnnType), cudaMemcpyDeviceToDevice);
	if (stepy2 > 1)
		VectorDiv << <GRIDSIZE, BLOCKSIZE >> > ((cudnnType*)C2.u, (cudnnType*)p+C1.paras.outSize, stepy2, C2.paras.outSize);
	else
		cudaMemcpy((cudnnType*)p+C1.paras.outSize, C2.u, C2.paras.outSize * sizeof(cudnnType), cudaMemcpyDeviceToDevice);
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

__global__ void adjustBasic(step_parameters steps, cudnnType*b, cudnnType *b_adj, int n)//scale n times
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int w_num, y_num;
	long long temp = b[tid] * 255;
	for (y_num = w_num = 0;y_num < n;y_num++)
	{
		while (w_num < n && (temp*steps.stepw[w_num]) / steps.stepw[w_num] == temp)
		{
			temp *= steps.stepw[w_num];
			w_num++;
		}
		if (temp > 0)
			temp = temp / steps.stepy[y_num] + (temp%steps.stepy[y_num] + steps.stepy[y_num] / 2) / steps.stepy[y_num];//avoid overflow
		else
			temp = temp / steps.stepy[y_num] + (temp%steps.stepy[y_num] - steps.stepy[y_num] / 2) / steps.stepy[y_num];//avoid overflow
	}
	b_adj[tid] = temp;
}
__global__ void adjustOutput(step_parameters steps, cudnnType*o, cudnnType *o_adj, int n)//scale n times
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int w_num, y_num;
	long long temp = o[tid];
	for (y_num = w_num = 0;w_num < n;w_num++)
	{
		while (y_num < n && (temp*steps.stepy[y_num]) / steps.stepy[y_num] == temp)
		{
			temp *= steps.stepy[y_num];
			y_num++;
		}
		if (temp > 0)
			temp = temp / steps.stepw[w_num] + (temp%steps.stepw[w_num] + steps.stepw[w_num] / 2) / steps.stepw[w_num];//avoid overflow
		else
			temp = temp / steps.stepw[w_num] + (temp%steps.stepw[w_num] - steps.stepw[w_num] / 2) / steps.stepw[w_num];//avoid overflow
	}
	o_adj[tid] = temp;
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