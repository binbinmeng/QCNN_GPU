/*
YUV data input and output
*/

#include <math.h>
typedef unsigned char datatype; //�˴�������8bit����λ��
typedef char restype;
typedef struct YChannel{
	//�漰��ͼ��������һ������height��width����yuv�洢��ʽͳһ
	int h;           // ͼ���
	int w;           // ͼ���
	int frames;		//֡��
	datatype* ImgData; // ͼ��������ά��̬����
}YChannel;

YChannel* get_Y(const char* filename, int frames, int height, int width); // ����ͼ��
