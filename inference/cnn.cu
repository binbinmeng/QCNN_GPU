#include "cnn.cuh"

CovLayer::CovLayer()
{
	//初始化张量描述符
	cudnnCreateFilterDescriptor(&wDesc);//初始化权重描述符
	cudnnCreateTensorDescriptor(&bDesc);
	cudnnCreateTensorDescriptor(&uDesc);
	cudnnCreateTensorDescriptor(&vDesc);
	cudnnCreateConvolutionDescriptor(&convDesc);//初始化卷积描述符
	cudnnCreateActivationDescriptor(&actiDesc);
}
int CovLayer::build(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int batch, int height, int width, int inChannel, int outChannel, int ksize)
{
	this->batch = batch;
	this->height = height;
	this->width = width;
	this->inChannel = inChannel;
	this->outChannel = outChannel;
	this->ksize = ksize;
	this->uSize = batch*outChannel*height*width;
	if (XWFORMAT == CUDNN_TENSOR_NCHW_VECT_C)
	{
		this->wSize = ksize*ksize*ceil((float)inChannel / 4) * 4 * outChannel;
		this->vSize = batch*ceil((float)outChannel / 4) * 4*height*width;
		check(cudnnSetFilter4dDescriptor(wDesc, XWTYPE, XWFORMAT, outChannel, ceil((float)inChannel / 4) * 4, ksize, ksize));
		check(cudnnSetTensor4dDescriptor(vDesc, XWFORMAT, XWTYPE, batch, ceil((float)outChannel / 4) * 4, height, width));
	}
	else
	{
		this->wSize = ksize*ksize*inChannel*outChannel;
		this->vSize = batch*outChannel*height*width;
		check(cudnnSetFilter4dDescriptor(wDesc, XWTYPE, XWFORMAT, outChannel, inChannel, ksize, ksize));
		check(cudnnSetTensor4dDescriptor(vDesc, XWFORMAT, XWTYPE, batch, outChannel, height, width));
	}
	check(cudnnSetTensor4dDescriptor(bDesc,
		BFORMAT,
		BTYPE,
		1, outChannel, 1, 1));
	check(cudnnSetTensor4dDescriptor(uDesc,
		YFORMAT,
		YTYPE,
		batch, outChannel, height, width));
	check(cudnnSetConvolution2dDescriptor(convDesc,
		(ksize - 1) / 2, (ksize - 1) / 2,//padding
		1, 1,//stride
		1, 1,//dilation
		CUDNN_CROSS_CORRELATION,
		CONVTYPE));
	check(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
		xDesc,
		wDesc,
		convDesc,
		uDesc,
		ALGO,
		&workspaceSize));
	check(cudnnSetActivationDescriptor(actiDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
	//build layer in GPU
	cudaDeviceSynchronize();
	check(cudaMalloc(&w, sizeof(xwtype)*this->wSize));
	check(cudaMalloc(&b, sizeof(btype)*outChannel));
	check(cudaMalloc(&b_adj, sizeof(btype)*outChannel));
	check(cudaMalloc(&u, sizeof(convtype)*this->uSize));
	check(cudaMalloc(&v, sizeof(xwtype)*this->vSize));
	return 0;
}

#if defined(INT8_EXT_CONFIG)||defined(INT8x4_EXT_CONFIG)
int CovLayer::load_para(FILE *fp)
{
	xwtype *w_h;
	btype *b_h;
	int *b_int, i;
	w_h = (xwtype*)malloc(sizeof(xwtype)*this->wSize);
	b_h = (btype*)malloc(sizeof(btype)*this->outChannel);
	b_int = (int*)malloc(sizeof(int)*this->outChannel);
	//convert format if necessary
	fread(&this->step_w, sizeof(int), 1, fp);
	fread(w_h, sizeof(xwtype), this->wSize, fp);
	fread(b_int, sizeof(int), this->outChannel, fp);
	for (i = 0;i < this->outChannel;i++)b_h[i] = b_int[i];
	check(cudaMemcpyAsync(w, w_h,	sizeof(xwtype) * this->wSize, cudaMemcpyHostToDevice));
	check(cudaMemcpyAsync(b, b_h, sizeof(btype) * this->outChannel, cudaMemcpyHostToDevice));
	//memdbg((convtype*)u, (xwtype*)v, (btype*)b, uSize);
	free(w_h);
	free(b_h);
	free(b_int);
	return 0;
}
int CovLayer::load_static_para(FILE *fp)//copy paras from file to GPU memory
{
	xwtype *w_h;
	btype *b_h;
	int *b_int, i;
	w_h = (xwtype*)malloc(sizeof(xwtype)*this->wSize);
	b_h = (btype*)malloc(sizeof(btype)*this->outChannel);
	b_int = (int*)malloc(sizeof(int)*this->outChannel);
	//convert format if necessary
	fread(w_h, sizeof(xwtype), this->wSize, fp);
	fread(b_int, sizeof(int), this->outChannel, fp);
	fread(&this->blu, sizeof(int), 1, fp);
	fread(&this->mul, sizeof(int), 1, fp);
	fread(&this->shift, sizeof(int), 1, fp);
	for (i = 0;i < this->outChannel;i++)b_h[i] = b_int[i];
	check(cudaMemcpyAsync(w, w_h, sizeof(xwtype) * this->wSize, cudaMemcpyHostToDevice));
	check(cudaMemcpyAsync(b, b_h, sizeof(btype) * this->outChannel, cudaMemcpyHostToDevice));
	//memdbg((convtype*)u, (xwtype*)v, (btype*)b, uSize);
	free(w_h);
	free(b_h);
	free(b_int);
	return 0;
}
#elif defined(FLOAT_CONFIG)
int CovLayer::load_para(FILE *fp)
{
	xwtype *w_h;
	btype *b_h;
	w_h = (xwtype*)malloc(sizeof(xwtype)*this->wSize);
	b_h = (btype*)malloc(sizeof(btype)*this->outChannel);
	fread(w_h, sizeof(xwtype), this->wSize, fp);
	fread(b_h, sizeof(btype), this->outChannel, fp);
	check(cudaMemcpyAsync(w, w_h, sizeof(xwtype) * this->wSize, cudaMemcpyHostToDevice));
	check(cudaMemcpyAsync(b, b_h, sizeof(btype) * this->outChannel, cudaMemcpyHostToDevice));
	free(w_h);
	free(b_h);
	return 0;
}
#endif
int CovLayer::ConvForward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void *x, void *workspace, size_t workspaceSize)//卷积
{
	cudaDeviceSynchronize();
	check(cudnnConvolutionForward(cudnnHandle, &alpha, xDesc,
		x, wDesc, w, convDesc,
		ALGO, workspace, workspaceSize, &beta,
		uDesc, u));//convolution
	cudaDeviceSynchronize();
	//convdbg((xwtype*)x, (xwtype*)w, (convtype*)u, (btype*)b);
#if defined(INT8_EXT_CONFIG)||defined(INT8x4_EXT_CONFIG)
	check(cudnnAddTensor(cudnnHandle, &alpha, bDesc, b_adj, &alpha, uDesc, u));//apply bias
#elif defined(FLOAT_CONFIG)
	check(cudnnAddTensor(cudnnHandle, &alpha, bDesc, b, &alpha, uDesc, u));//apply bias
#endif
	return 0;
}
int CovLayer::ConvForward_static(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void *x, void *workspace, size_t workspaceSize)//卷积
{
	cudaDeviceSynchronize();
	check(cudnnConvolutionForward(cudnnHandle, &alpha, xDesc,
		x, wDesc, w, convDesc,
		ALGO, workspace, workspaceSize, &beta,
		uDesc, u));//convolution
	cudaDeviceSynchronize();
	//convdbg((xwtype*)x, (xwtype*)w, (convtype*)u, (btype*)b);
#if defined(INT8_EXT_CONFIG)||defined(INT8x4_EXT_CONFIG)
	check(cudnnAddTensor(cudnnHandle, &alpha, bDesc, b, &alpha, uDesc, u));//apply bias
#elif defined(FLOAT_CONFIG)
	check(cudnnAddTensor(cudnnHandle, &alpha, bDesc, b, &alpha, uDesc, u));//apply bias
#endif
	return 0;
}
int CovLayer::activate(cudnnHandle_t cudnnHandle)
{
	cudaDeviceSynchronize();
	check(cudnnActivationForward(cudnnHandle, actiDesc, &alpha, uDesc, u, &beta, uDesc, u));
	return 0;
}

#if defined(INT8_EXT_CONFIG)||defined(INT8x4_EXT_CONFIG)
int CovLayer::quantize_out(void*workspace)//u(float)->mid(int)->y(char)
{
	convtype max;
	cudaDeviceSynchronize();
	findMax((convtype*)u, (convtype*)workspace, this->uSize, &max);
	max_u = max;
	step_y = max / (THRESHOLD+1) + 1;
	NCHW2NCHW_VECT_C((convtype*)u, (xwtype*)v, step_y, batch, outChannel, height, width);
	return 0;
}
int CovLayer::quantize_out_fix(int max_u)
{
	convtype max;
	cudaDeviceSynchronize();
	max = max_u;
	this->max_u = max_u;
	step_y = max / (THRESHOLD + 1) + 1;
	NCHW2NCHW_VECT_C((convtype*)u, (xwtype*)v, step_y, batch, outChannel, height, width);
	return 0;
}
int CovLayer::quantize_out_static(void)//static_layer
{
	cudaDeviceSynchronize();
	mul_shift<<<GRIDSIZE,BLOCKSIZE>>>((convtype*)u, (xwtype*)v, this->uSize, this->mul, this->shift);
	return 0;
}
int CovLayer::quantize_out_blu(void)//static_layer
{
	cudaDeviceSynchronize();
	NCHW2NCHW_VECT_C_QUANT_BLU((convtype*)u, (xwtype*)v, batch, outChannel, height, width, blu, mul, shift);
	return 0;
}
#endif

int CovLayer::viewmem(xwtype*x)
{
	int i,j;
	xwtype*x_h=new xwtype[height*width*inChannel*batch];
	xwtype*w_h = new xwtype[ksize*ksize*inChannel*batch];
	convtype *u_h = new convtype[uSize];
	xwtype *v_h = new xwtype[vSize];
	btype b_h[16],b_adj_h[16];
	cudaMemcpy(x_h, x, sizeof(xwtype)*height*width*inChannel*batch, cudaMemcpyDeviceToHost);
	cudaMemcpy(w_h, w, sizeof(xwtype)*ksize*ksize*inChannel*batch, cudaMemcpyDeviceToHost);
	cudaMemcpy(u_h, u, sizeof(convtype)*uSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(v_h, v, sizeof(xwtype)*vSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(b_h, b, sizeof(btype) * 16, cudaMemcpyDeviceToHost);
	cudaMemcpy(b_adj_h, b_adj, sizeof(btype) * 16, cudaMemcpyDeviceToHost);
	printf("x:\n");
	for (i = 0;i < 5;i++)
	{
		for (j = 0;j < 5;j++)
			printf("%d	", x_h[(i*width + j) * 4]);
		printf("\n");
	}
	printf("weights:\n");
	for (i = 0;i < 5;i++)
		printf("%d	", w_h[i * 4]);
	printf("\nbiases:%f\n", b_h[0]);
	printf("u:\n");
	for (i = 0;i < 5;i++)
	{
		for (j = 0;j < 5;j++)
			printf("%f	", u_h[i*width + j]);
		printf("\n");
	}
	printf("mul:%d,shift:%d\n",mul,shift);
	printf("v:\n");
	for (i = 0;i < 5;i++)
	{
		for (j = 0;j < 5;j++)
			printf("%d	", v_h[(i*width + j)*4]);
		printf("\n");
	}
	free(x_h);
	free(w_h);
	free(u_h);
	free(v_h);
	return 0;
}

CovLayer::~CovLayer()
{
	cudnnDestroyFilterDescriptor(wDesc);
	cudnnDestroyTensorDescriptor(bDesc);
	cudnnDestroyTensorDescriptor(uDesc);
	cudnnDestroyTensorDescriptor(vDesc);
	cudnnDestroyConvolutionDescriptor(convDesc);
	cudnnDestroyActivationDescriptor(actiDesc);
	cudaFree(w);
	cudaFree(b);
	cudaFree(b_adj);
	cudaFree(u);
	cudaFree(v);
}
ConcatLayer::ConcatLayer(void)
{
	cudnnCreateTensorDescriptor(&concDesc);
}
int ConcatLayer::build(int batch, int height, int width, int inChannel1, int inChannel2)
{
	this->batch = batch;
	this->height = height;
	this->width = width;
	this->inChannel1 = inChannel1;
	this->inChannel2 = inChannel2;
	this->outChannel = inChannel1 + inChannel2;
	check(cudnnSetTensor4dDescriptor(concDesc,
		XWFORMAT,
		XWTYPE,
		batch, outChannel, height, width));
	cudaDeviceSynchronize();
	check(cudaMalloc(&conc, sizeof(xwtype)*batch*outChannel*height*width));
	return 0;
}
#if defined(INT8_EXT_CONFIG)||defined(INT8x4_EXT_CONFIG)
int ConcatLayer::concat(CovLayer *C1, CovLayer *C2, void *workspace)
{
	int i;
	convtype max1, max2;
	int stepy1, stepy2;
	cudaDeviceSynchronize();
	findMax((convtype*)C1->u, (convtype*)workspace, C1->uSize, &max1);
	findMax((convtype*)C2->u, (convtype*)workspace, C2->uSize, &max2);
	C1->max_u = max1;
	C2->max_u = max2;
	if (max1 > THRESHOLD)
		stepy1 = max1 / (THRESHOLD + 1) + 1;
	else
		stepy1 = 1;
	if (max2 > THRESHOLD)
		stepy2 = max2 / (THRESHOLD + 1) + 1;
	else
		stepy2 = 1;
	//对stepy进行调整以保持ratio一致,结果不能太小
	if (C1->step_w*stepy2 > C2->step_w*stepy1)
		stepy1 = (C1->step_w*stepy2 + (C2->step_w >> 1)) / C2->step_w;
	else
		stepy2 = (C2->step_w*stepy1 + (C1->step_w >> 1)) / C1->step_w;
	C1->step_y = stepy1;
	C2->step_y = stepy2;

	//这是一个安全隐患，当C1.outChannel%4!=0时会出错
	for (i = 0;i < batch;i++)
	{
		CHW2CHW_VECT_C <<<GRIDSIZE, BLOCKSIZE>>>((convtype*)C1->u + i*inChannel1*height*width, (xwtype*)conc + i*outChannel*height*width, stepy1, height*width,inChannel1,GRIDSIZE*BLOCKSIZE);
		CHW2CHW_VECT_C <<<GRIDSIZE, BLOCKSIZE>>>((convtype*)C2->u + i*inChannel2*height*width, (xwtype*)conc + i*outChannel*height*width + inChannel1*height*width, stepy2, height*width, inChannel2, GRIDSIZE*BLOCKSIZE);
		//NCHW2NCHW_VECT_C((convtype*)C1->u, (xwtype*)conc, stepy1, C1->batch, C1->outChannel, C1->height, C1->width);
		//NCHW2NCHW_VECT_C((convtype*)C2->u, (xwtype*)conc + C1->vSize, stepy2, C2->batch, C2->outChannel, C2->height, C2->width);
	}
	return 0;
}
int ConcatLayer::concat_fix(CovLayer *C1, CovLayer *C2, convtype max1, convtype max2)
{
	int i;
	int stepy1, stepy2;
	cudaDeviceSynchronize();
	C1->max_u = max1;
	C2->max_u = max2;
	if (max1 > THRESHOLD)
		stepy1 = max1 / (THRESHOLD + 1) + 1;
	else
		stepy1 = 1;
	if (max2 > THRESHOLD)
		stepy2 = max2 / (THRESHOLD + 1) + 1;
	else
		stepy2 = 1;
	//对stepy进行调整以保持ratio一致,结果不能太小
	if (C1->step_w*stepy2 > C2->step_w*stepy1)
		stepy1 = (C1->step_w*stepy2 + (C2->step_w >> 1)) / C2->step_w;
	else
		stepy2 = (C2->step_w*stepy1 + (C1->step_w >> 1)) / C1->step_w;
	C1->step_y = stepy1;
	C2->step_y = stepy2;

	//这是一个安全隐患，当C1.outChannel%4!=0时会出错
	for (i = 0;i < batch;i++)
	{
		CHW2CHW_VECT_C << <GRIDSIZE, BLOCKSIZE >> >((convtype*)C1->u + i*inChannel1*height*width, (xwtype*)conc + i*outChannel*height*width, stepy1, height*width, inChannel1, GRIDSIZE*BLOCKSIZE);
		CHW2CHW_VECT_C << <GRIDSIZE, BLOCKSIZE >> >((convtype*)C2->u + i*inChannel2*height*width, (xwtype*)conc + i*outChannel*height*width + inChannel1*height*width, stepy2, height*width, inChannel2, GRIDSIZE*BLOCKSIZE);
		//NCHW2NCHW_VECT_C((convtype*)C1->u, (xwtype*)conc, stepy1, C1->batch, C1->outChannel, C1->height, C1->width);
		//NCHW2NCHW_VECT_C((convtype*)C2->u, (xwtype*)conc + C1->vSize, stepy2, C2->batch, C2->outChannel, C2->height, C2->width);
	}
	return 0;
}
/*
int ConcatLayer::concat_static(CovLayer *C1, CovLayer *C2)
{
	int i, frameSize, frameSize1, frameSize2, channelSize1, channelSize2;
	C1->step_y = ((1 << C1->shift) + C1->mul / 2) / C1->mul;
	C2->step_y = ((1 << C2->shift) + C2->mul / 2) / C2->mul;
	C1->max_u = 127 * C1->step_y;
	C2->max_u = 127 * C2->step_y;
	frameSize1 = C1->outChannel*C1->height*C1->width;
	channelSize1 = C1->height*C1->width;
	frameSize2 = C2->outChannel*C2->height*C2->width;
	channelSize2 = C2->height*C2->width;
	frameSize = frameSize1 + frameSize2;
	cudaDeviceSynchronize();
	for (i = 0;i < C1->batch;i++)
	{
		CHW2CHW_VECT_C_QUANT <<<GRIDSIZE, BLOCKSIZE>>> ((convtype*)C1->u + i*frameSize1, (xwtype*)conc + i*frameSize, channelSize1, C1->outChannel, GRIDSIZE*BLOCKSIZE, C1->mul, C1->shift);
		CHW2CHW_VECT_C_QUANT <<<GRIDSIZE, BLOCKSIZE>>> ((convtype*)C2->u + i*frameSize2, (xwtype*)conc + i*frameSize+frameSize1, channelSize2, C2->outChannel, GRIDSIZE*BLOCKSIZE, C2->mul, C2->shift);
	}
	return 0;
}*/
int ConcatLayer::concat_blu(CovLayer *C1, CovLayer *C2)
{
	int i, frameSize, frameSize1, frameSize2, channelSize1, channelSize2;
	C1->step_y = ((1 << C1->shift) + C1->mul / 2) / C1->mul;
	C2->step_y = ((1 << C2->shift) + C2->mul / 2) / C2->mul;
	C1->max_u = 127 * C1->step_y;
	C2->max_u = 127 * C2->step_y;
	frameSize1 = C1->outChannel*C1->height*C1->width;
	channelSize1 = C1->height*C1->width;
	frameSize2 = C2->outChannel*C2->height*C2->width;
	channelSize2 = C2->height*C2->width;
	frameSize = frameSize1 + frameSize2;
	cudaDeviceSynchronize();
	for (i = 0;i < C1->batch;i++)
	{
		CHW2CHW_VECT_C_QUANT_BLU << <GRIDSIZE, BLOCKSIZE >> > ((convtype*)C1->u + i*frameSize1, (xwtype*)conc + i*frameSize, channelSize1, C1->outChannel, GRIDSIZE*BLOCKSIZE, C1->blu, C1->mul, C1->shift);
		CHW2CHW_VECT_C_QUANT_BLU << <GRIDSIZE, BLOCKSIZE >> > ((convtype*)C2->u + i*frameSize2, (xwtype*)conc + i*frameSize + frameSize1, channelSize2, C2->outChannel, GRIDSIZE*BLOCKSIZE, C2->blu, C2->mul, C2->shift);
	}
	return 0;
}
#elif defined(FLOAT_CONFIG)
int ConcatLayer::concat(CovLayer *C1, CovLayer *C2, void *workspace)
{
	cudaDeviceSynchronize();
	cudaMemcpy(conc, C1->u, C1->uSize*sizeof(convtype), cudaMemcpyDeviceToDevice);
	cudaMemcpy((convtype*)conc+C1->uSize, C2->u, C2->uSize*sizeof(convtype), cudaMemcpyDeviceToDevice);
	return 0;
}
#endif
ConcatLayer::~ConcatLayer(void)
{
	cudnnDestroyTensorDescriptor(concDesc);
	cudaFree(conc);
}

InputLayer::InputLayer(void)
{
	cudnnCreateTensorDescriptor(&xDesc);
}
int InputLayer::build(int batch, int channel, int height, int width)
{
	this->batch = batch;
	this->height = height;
	this->width = width;
	this->inChannel = channel;
	this->inSize = batch*height*width*channel;
	if (XWTYPE == CUDNN_DATA_INT8x4)
	{
		outChannel = ceil((float)channel / 4) * 4;
		outSize = batch*height*width*outChannel;
	}
	else if (XWTYPE == CUDNN_DATA_FLOAT)
	{
		outChannel = channel;
		outSize = inSize;
	}
	cudaDeviceSynchronize();
	check(cudnnSetTensor4dDescriptor(xDesc, XWFORMAT, XWTYPE, batch, outChannel, height, width));
	check(cudaMalloc(&x, sizeof(datatype)*inSize));
	check(cudaMalloc(&x_rec, sizeof(datatype)*inSize));
	check(cudaMalloc(&x_ppro, sizeof(xwtype)*outSize));
	//check(cudaMemset(x_ppro,0, sizeof(xwtype)*outSize));
	return 0;
}
int InputLayer::load(datatype *input)
{
	cudaMemcpy(x, input, sizeof(datatype)*batch*inChannel*height*width, cudaMemcpyHostToDevice);
	return 0;
}
#if defined(INT8x4_EXT_CONFIG)
__global__ void HW2HW_VECT_C_PPRO(datatype*x, xwtype*x_ppro, int n, int gridSize)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	while (tid < n)
	{
		x_ppro[tid * 4] = x[tid] - 128;
		tid += gridSize;
	}
}
int InputLayer::ppro(void)
{
	int i, j, c, cv;
	int frameSize, channelSize;
	frameSize = height*width*inChannel;
	channelSize = height*width;
	for (i = 0;i<batch;i++)
		for (j = 0;j < inChannel;j++)
		{
			c = j >> 2;
			cv = j & 3;
			HW2HW_VECT_C_PPRO << <GRIDSIZE, BLOCKSIZE >> > ((datatype*)x + i*frameSize + j*channelSize, (xwtype*)x_ppro + i*frameSize + c*channelSize * 4 + cv,height*width,GRIDSIZE*BLOCKSIZE);
		}
	return 0;
}
#elif defined(FLOAT_CONFIG)
__global__ void FLOAT_PPRO(datatype*x, xwtype*x_ppro,int num,int gridSize)
{
	int i;
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	while (tid < num)
	{
		x_ppro[tid] = ((float)x[tid] - 128) / 255;
		tid += gridSize;
	}
}
int InputLayer::ppro(void)
{
	FLOAT_PPRO <<<GRIDSIZE, BLOCKSIZE>>>((datatype*)x, (xwtype*)x_ppro, outSize, GRIDSIZE*BLOCKSIZE);
	return 0;
}
#endif

__global__ void applyRes_GPU(datatype*x, xwtype*res, datatype*rec, int num, int gridSize)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	short int rec_int;
	while (tid < num)
	{
#if defined(INT8_EXT_CONFIG)||defined(INT8x4_EXT_CONFIG)
		rec_int = (short int)x[tid] + res[tid];
#elif defined(FLOAT_CONFIG)
		rec_int = x[tid] + res[tid] * 255 + 0.5;
#endif
		if (rec_int > 255)rec[tid] = 255;
		else if (rec_int < 0)rec[tid] = 0;
		else rec[tid] = rec_int;
		tid += gridSize;
	}
	/*
	if(x[tid]>CLIP_UPPER || x[tid] < CLIP_LOWER)rec[tid] = x[tid];
	else rec[tid] = x[tid]+res[tid];*/
}
__global__ void applyRes_GPU_y(datatype*x, datatype*rec, convtype*res, int mul, int shift, int num, int gridSize)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	short int rec_int;
	int bias, temp;
	bias = 1 << shift - 1;
	while (tid < num)
	{
		temp = res[tid];
		temp = temp * mul + bias >> shift;
		rec_int = (short int)x[tid] + temp;
		if (rec_int > 255)rec[tid] = 255;
		else if (rec_int < 0)rec[tid] = 0;
		else rec[tid] = rec_int;
		tid += gridSize;
	}
}
int InputLayer::applyRes(xwtype*res)
{
	cudaDeviceSynchronize();
	applyRes_GPU <<<BLOCKSIZE, GRIDSIZE>>> ((datatype*)x, res, (datatype*)x_rec,inSize,BLOCKSIZE*GRIDSIZE);
	return 0;
}
int InputLayer::applyRes_y(convtype*res, int mul, int shift)
{
	cudaDeviceSynchronize();
	applyRes_GPU_y << <BLOCKSIZE, GRIDSIZE >> > ((datatype*)x, (datatype*)x_rec, res, mul, shift, inSize, BLOCKSIZE*GRIDSIZE);
	return 0;
}
int InputLayer::viewmem(xwtype*res)
{
	int nsize = height*width*inChannel*batch;
	xwtype*res_h = new xwtype[nsize];
	datatype*x_h = new datatype[nsize];
	datatype*rec_h = new datatype[nsize];
	cudaMemcpy(res_h, res, sizeof(xwtype)*nsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(x_h, x, sizeof(datatype)*nsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(rec_h, x_rec, sizeof(datatype)*nsize, cudaMemcpyDeviceToHost);
	free(res_h);
	free(x_h);
	free(rec_h);
	return 0;
}
InputLayer::~InputLayer(void)
{
	cudnnDestroyTensorDescriptor(xDesc);
	cudaFree(x);
	cudaFree(x_ppro);
	cudaFree(x_rec);
}
