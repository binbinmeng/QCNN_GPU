#include "yuv_data.h"

vrcnn_data::vrcnn_data(int frame, int height, int width)
{
	this->frame = frame;
	this->h = height;
	this->w = width;
	this->nSize = frame*height*width;
	this->ori = new datatype[nSize];
	this->input = new datatype[nSize];
	this->recon = new datatype[nSize];
	this->norm = new restype[nSize];
	this->res = new restype[nSize];
}
int vrcnn_data::read_data(char *orifile, char *inputfile)
{
	int i;
	FILE *orifp, *inputfp;
	if (fopen_s(&orifp, orifile, "rb"))
	{
		printf("%s\n", orifile);
		printf("open ori file failed\n");
		exit(1);
	}
	if (fopen_s(&inputfp, inputfile, "rb"))
	{
		printf("%s\n", inputfile);
		printf("open input file failed\n");
		fclose(orifp);
		exit(1);
	}
	for (i = 0; i < frame; ++i)
	{
		fread(this->ori + i*h*w, sizeof(datatype), h*w, orifp);
		fread(this->input + i*h*w, sizeof(datatype), h*w, inputfp);
		fseek(orifp, sizeof(datatype)*h*w / 2, SEEK_CUR);
		fseek(inputfp, sizeof(datatype)*h*w / 2, SEEK_CUR);
	}
	fclose(orifp);
	fclose(inputfp);
	return 0;
}

int vrcnn_data::read_frame(char *orifile, char *inputfile, int n)
{
	FILE *orifp, *inputfp;
	frame = 1;
	if (fopen_s(&orifp, orifile, "rb"))
	{
		printf("open file failed\n");
		return 1;
	}
	if (fopen_s(&inputfp, inputfile, "rb"))
	{
		printf("open file failed\n");
		fclose(orifp);
		return 1;
	}
	fseek(orifp, sizeof(datatype)*h*w*n * 3 / 2, SEEK_CUR);
	fread(ori, sizeof(datatype), h*w, orifp);
	fseek(inputfp, sizeof(datatype)*h*w*n * 3 / 2, SEEK_CUR);
	fread(input, sizeof(datatype), h*w, inputfp);
	fclose(orifp);
	fclose(inputfp);
	return 0;
}
int vrcnn_data::preprocess(void)//已改为在GPU中处理
{
	int i;
	for (i = 0;i < nSize;i++)
		norm[i] = (int)input[i] - 128;
	return 0;
}
int vrcnn_data::loadRes_GPU(xwtype*v)
{
	cudaDeviceSynchronize();
	cudaMemcpy(res, v, nSize, cudaMemcpyDeviceToHost);
	return 0;
}
int vrcnn_data::applyRes(void)
{
	int i;
	for (i = 0;i < nSize;i++)
		recon[i] = (int)input[i] + res[i];
	return 0;
}
double vrcnn_data::psnr(datatype*data)
{
	int i;
	double mse, psnr;
	mse = 0;
	for (i = 0; i < nSize; i++)
		mse += (data[i] - ori[i])*(data[i] - ori[i]);
	mse /= nSize;
	psnr = 10 * log10(65025.0 / mse);
	return psnr;
}
double vrcnn_data::psnr_pf(void)
{
	int i, n;
	double mse, psnr;
	for (n = 0;n < frame;n++)
	{
		mse = 0;
		for (i = 0; i < h*w; i++)
			mse += (recon[n*h*w+i] - ori[n*h*w+i])*(recon[n*h*w+i] - ori[n*h*w+i]);
		mse /= h*w;
		psnr = 10 * log10(65025.0 / mse);
		printf("PSNR of Frame %d:%f\n", n, psnr);
	}
	return psnr;
}
int vrcnn_data::save_recon_as(char* filename)
{
	int i;
	FILE  *fp;
	if (fopen_s(&fp, filename, "wb"))
		printf("write file failed\n");
	datatype *uv = new datatype[h*w / 2];
	memset(uv, 0, h*w / 2);
	for (i = 0; i < frame; i++)
	{
		fwrite(recon + i * h * w, sizeof(datatype), h*w, fp);
		fwrite(uv, sizeof(datatype), h * w / 2, fp);
	}
	fclose(fp);
	return 0;
}
vrcnn_data::~vrcnn_data(void)
{
	free(ori);
	free(input);
	free(recon);
	free(norm);
	free(res);
}
