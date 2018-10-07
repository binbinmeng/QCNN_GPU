#include <stdlib.h>
#include <stdio.h>
#include "yuv_data.h"

YChannel* get_Y(const char* filename, int frames, int height, int width) // 读入图像
{
	// 图像数组的初始化
	YChannel* data_Y = new YChannel;

	if (filename == NULL)
	{
		data_Y->ImgData = (unsigned char*)malloc(sizeof(char)*frames*height*width);
		data_Y->frames = frames;
		data_Y->h = height;
		data_Y->w = width;
		return data_Y;
	}

	FILE  *fp = NULL;
	if(fopen_s(&fp, filename, "rb"))
		printf("open file failed\n");

	data_Y->ImgData = (unsigned char*)malloc(sizeof(char)*frames*height*width);
	int i;
	for (i = 0; i < frames; ++i)
	{
		fread(data_Y->ImgData+i*height*width, sizeof(datatype), height*width, fp);
		fseek(fp, height*width / 2, SEEK_CUR);
	}
	fclose(fp);
	data_Y->frames = frames;
	data_Y->h = height;
	data_Y->w = width;
	return data_Y;
}

