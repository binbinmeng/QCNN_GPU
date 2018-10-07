/*
YUV data input and output
*/

#include <math.h>
typedef unsigned char datatype; //此处适用于8bit像素位宽
typedef char restype;
typedef struct YChannel{
	//涉及到图像类数据一律是先height后width，与yuv存储格式统一
	int h;           // 图像高
	int w;           // 图像宽
	int frames;		//帧数
	datatype* ImgData; // 图像数据三维动态数组
}YChannel;

YChannel* get_Y(const char* filename, int frames, int height, int width); // 读入图像
