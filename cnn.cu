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
	check(cudaMalloc(&w, sizeof(xwtype)*this->wSize));
	check(cudaMalloc(&b, sizeof(btype)*outChannel));
	check(cudaMalloc(&b_adj, sizeof(btype)*outChannel));
	check(cudaMalloc(&u, sizeof(convtype)*this->uSize));
	check(cudaMalloc(&v, sizeof(xwtype)*this->vSize));
	return 0;
}

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

int CovLayer::ConvForward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void *x, void *workspace, size_t workspaceSize)//卷积
{
	cudaDeviceSynchronize();
	check(cudnnConvolutionForward(cudnnHandle, &alpha, xDesc,
		x, wDesc, w, convDesc,
		ALGO, workspace, workspaceSize, &beta,
		uDesc, u));//convolution
	cudaDeviceSynchronize();
	//convdbg((xwtype*)x, (xwtype*)w, (convtype*)u, (btype*)b);
	check(cudnnAddTensor(cudnnHandle, &alpha, bDesc, b_adj, &alpha, uDesc, u));//apply bias
	return 0;
}
int CovLayer::activate(cudnnHandle_t cudnnHandle)
{
	cudaDeviceSynchronize();
	check(cudnnActivationForward(cudnnHandle, actiDesc, &alpha, uDesc, u, &beta, uDesc, u));
	return 0;
}
int CovLayer::quantize_out(void*workspace)//u(float)->mid(int)->y(char)
{
	convtype max;
	cudaDeviceSynchronize();
	findMax((convtype*)u, (convtype*)workspace, this->uSize, &max);
	this->max_u = max;
	step_y = max / (THRESHOLD+1) + 1;
	NCHW2NCHW_VECT_C((convtype*)u, (xwtype*)v, step_y, batch, outChannel, height, width);
	return 0;
}

int CovLayer::viewmem(xwtype*x)
{
	xwtype*x_h=new xwtype[height*width*inChannel*batch];
	convtype *u_h = new float[uSize];
	xwtype *v_h = new char[vSize];
	btype b_h[16],b_adj_h[16];
	cudaMemcpy(x_h, x, sizeof(xwtype)*height*width*inChannel*batch, cudaMemcpyDeviceToHost);
	cudaMemcpy(u_h, u, sizeof(convtype)*uSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(v_h, v, sizeof(xwtype)*vSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(b_h, b, sizeof(btype) * 16, cudaMemcpyDeviceToHost);
	cudaMemcpy(b_adj_h, b_adj, sizeof(btype) * 16, cudaMemcpyDeviceToHost);
	free(x_h);
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
	check(cudaMalloc(&conc, sizeof(xwtype)*batch*outChannel*height*width));
	return 0;
}
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
	//这是一个安全隐患，当batch!=1时会出错
	for (i = 0;i < batch;i++)
	{
		CHW2CHW_VECT_C <<<HEIGHT, WIDTH>>>((convtype*)C1->u + i*inChannel1*height*width, (xwtype*)conc + i*outChannel*height*width, stepy1, inChannel1);
		CHW2CHW_VECT_C <<<HEIGHT, WIDTH>>>((convtype*)C2->u + i*inChannel2*height*width, (xwtype*)conc + i*outChannel*height*width + inChannel1*height*width, stepy2, inChannel2);
		//NCHW2NCHW_VECT_C((convtype*)C1->u, (xwtype*)conc, stepy1, C1->batch, C1->outChannel, C1->height, C1->width);
		//NCHW2NCHW_VECT_C((convtype*)C2->u, (xwtype*)conc + C1->vSize, stepy2, C2->batch, C2->outChannel, C2->height, C2->width);
	}
	return 0;
}
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

	check(cudaMalloc(&x, sizeof(datatype)*inSize));
	check(cudaMalloc(&x_rec, sizeof(datatype)*inSize));
	if (XWFORMAT == CUDNN_TENSOR_NCHW_VECT_C)
	{
		outChannel = ceil((float)channel / 4) * 4;
		outSize = batch*height*width*outChannel;
		check(cudnnSetTensor4dDescriptor(xDesc, XWFORMAT, XWTYPE, batch, outChannel, height, width));
		check(cudaMalloc(&x_ppro, sizeof(xwtype)*outSize));
		//check(cudaMemset(x_ppro,0, sizeof(xwtype)*outSize));
	}
	else
	{
		printf("undefined input format!\n");
		exit(1);
	}
	return 0;
}
int InputLayer::load(datatype *input)
{
	cudaMemcpy(x, input, sizeof(datatype)*batch*height*width, cudaMemcpyHostToDevice);
	return 0;
}
__global__ void HW2HW_VECT_C_PPRO(datatype*x, xwtype*x_ppro)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	x_ppro[tid * 4] = x[tid] - 128;
}
int InputLayer::ppro(void)
{
	int i, j, c, cv;
	int frameSize, channelSize;
	frameSize = height*width*inChannel;
	channelSize = height*width;
	if (XWTYPE == CUDNN_DATA_INT8x4)
	{
		for(i=0;i<batch;i++)
			for (j = 0;j < inChannel;j++)
			{
				c = j >> 2;
				cv = j & 3;
				HW2HW_VECT_C_PPRO <<<HEIGHT, WIDTH>>> ((datatype*)x + i*frameSize + j*channelSize, (xwtype*)x_ppro + i*frameSize + c*channelSize * 4 + cv);
			}
		return 0;
	}
	else
	{
		printf("undefined input format!\n");
		exit(1);
	}
}
__global__ void applyRes_GPU(datatype*x, xwtype*res, datatype*rec)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	rec[tid] = x[tid] + res[tid];
}
int InputLayer::applyRes(xwtype*res)
{
	applyRes_GPU <<<HEIGHT, WIDTH>>> ((datatype*)x, res, (datatype*)x_rec);
	return 0;
}
InputLayer::~InputLayer(void)
{
	cudnnDestroyTensorDescriptor(xDesc);
	cudaFree(x);
}
