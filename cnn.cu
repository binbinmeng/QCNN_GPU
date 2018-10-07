#include "cnn.cuh"

CovLayer::CovLayer()
{
	//初始化张量描述符
	cudnnCreateFilterDescriptor(&wDesc);//初始化权重描述符
	cudnnCreateTensorDescriptor(&bDesc);
	cudnnCreateTensorDescriptor(&uDesc);
	cudnnCreateTensorDescriptor(&yDesc);
	cudnnCreateConvolutionDescriptor(&convDesc);//初始化卷积描述符
}
int CovLayer::build(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int inChannel, int outChannel, int height, int width, int ksize)
{


	check(cudnnSetFilter4dDescriptor(wDesc,
		CUDNNTYPE,
		CUDNN_TENSOR_NCHW,
		outChannel, inChannel, ksize, ksize));
	check(cudnnSetTensor4dDescriptor(bDesc,
		CUDNN_TENSOR_NCHW,
		CUDNNTYPE,
		1, outChannel, 1, 1));
	check(cudnnSetTensor4dDescriptor(uDesc,
		CUDNN_TENSOR_NCHW,
		CUDNNTYPE,
		1, outChannel, height, width));
	check(cudnnSetTensor4dDescriptor(yDesc,
		CUDNN_TENSOR_NCHW,
		CUDNNTYPE,
		1, outChannel, height, width));
	check(cudnnSetConvolution2dDescriptor(convDesc,
		(ksize - 1) / 2, (ksize - 1) / 2,//padding
		1, 1,//stride
		1, 1,//dilation
		CUDNN_CROSS_CORRELATION,
		CUDNNTYPE));
	/*check(cudnnGetConvolution2dForwardOutputDim(convDesc,
	xDesc,
	wDesc,
	&batch, &channel, &height, &width));*/
	//cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &return_value);//I know it's 8
	check(cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle,
		xDesc,
		wDesc,
		convDesc,
		uDesc,
		8,
		&algo_num,
		perfResults));//转移到CovFwd
	check(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
		xDesc,
		wDesc,
		convDesc,
		uDesc,
		perfResults[0].algo,
		&paras.workspaceSize));

	paras.inChannel = inChannel;
	paras.outChannel = outChannel;
	paras.ksize = ksize;
	paras.height = height;
	paras.width = width;
	paras.outSize = outChannel*height*width;

	//build layer in GPU
	check(cudaMalloc(&w, sizeof(cudnnType)*outChannel*inChannel*ksize*ksize));
	check(cudaMalloc(&b, sizeof(cudnnType)*outChannel));
	if (outChannel*height*width > GRIDSIZE*BLOCKSIZE * 2)
	{
		check(cudaMalloc(&u, sizeof(cudnnType)*outChannel*height*width));
		check(cudaMalloc(&y, sizeof(cudnnType)*outChannel*height*width));
	}
	else
	{
		check(cudaMalloc(&u, sizeof(cudnnType)*GRIDSIZE*BLOCKSIZE * 2));
		check(cudaMalloc(&y, sizeof(cudnnType)*GRIDSIZE*BLOCKSIZE * 2));
	}
	return 0;
}

int CovLayer::load_para(FILE *fp)
{
	int w_num = paras.inChannel*paras.outChannel*paras.ksize*paras.ksize;
	int b_num = paras.outChannel;
	char *w_ori, *b_ori;
	int *w_cvt, *b_cvt;
	w_ori = (char*)malloc(sizeof(char)*w_num);
	b_ori = (char*)malloc(sizeof(char)*b_num);
	w_cvt = (int*)malloc(sizeof(int)*w_num);
	b_cvt = (int*)malloc(sizeof(int)*b_num);
	//convert format if necessary
	fread(w_ori, sizeof(char), w_num, fp);
	fread(b_ori, sizeof(char), b_num, fp);
	for (int i = 0;i < w_num;i++)w_cvt[i] = w_ori[i];
	for (int i = 0;i < b_num;i++)b_cvt[i] = b_ori[i];
	check(cudaMemcpyAsync(w, w_cvt,
		sizeof(cudnnType) * paras.inChannel*paras.outChannel*paras.ksize*paras.ksize, cudaMemcpyHostToDevice));
	check(cudaMemcpyAsync(b, b_cvt,
		sizeof(cudnnType) * paras.outChannel, cudaMemcpyHostToDevice));
	return 0;
}

int CovLayer::ConvForward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void *x, void *workspace, size_t workspaceSize)//卷积
{

	check(cudnnConvolutionForward(cudnnHandle, &paras.alpha, xDesc,
		x, wDesc, w, convDesc,
		perfResults[0].algo, workspace, workspaceSize, &paras.beta,
		uDesc, u));//convolution
	check(cudnnAddTensor(cudnnHandle, &paras.alpha, bDesc,
		b, &paras.alpha, uDesc, u));//apply bias
	return 0;
}
int CovLayer::quantize_out(void *workspace)
{
	cudnnType max;
	int i, uSize;
	uSize = paras.outChannel*paras.height*paras.width;
	if (uSize > GRIDSIZE*BLOCKSIZE * 2)
		findMax((cudnnType*)u, (cudnnType*)workspace, uSize, &max);
	else
		findMax((cudnnType*)u, (cudnnType*)workspace, GRIDSIZE*BLOCKSIZE * 2, &max);
	if (max > THRESHOLD)
	{
		step_y = max / (THRESHOLD+1) + 1;
		VectorDiv << <GRIDSIZE, BLOCKSIZE >> > ((cudnnType*)u, (cudnnType*)y, step_y, uSize);
	}
	else
	{
		step_y = 1;
		cudaMemcpy(y, u, sizeof(cudnnType)*uSize, cudaMemcpyDeviceToDevice);
	}
	return 0;
}
size_t CovLayer::get_result(void *out)//获取卷积结果
{
	size_t u_size = paras.outChannel*paras.height*paras.width;
	check(cudaMemcpy(out, u, sizeof(cudnnType) * u_size, cudaMemcpyDeviceToHost));
	return u_size;
}