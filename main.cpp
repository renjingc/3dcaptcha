#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

//temp矩阵中0为初始化，1为边界字符，2为基准线像素,3为字符表面
void initTemp(Mat& src)
{
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			src.at<uchar>(i, j) = 0;
		}
	}
}
//选择初步文本像素
//提取字符边界
void selectTextFirst(Mat& src, Mat& dst,Mat& temp)
{
	for (int i = 1; i < src.rows-1; i++)
	{
		const uchar * previous = src.ptr<const uchar>(i - 1); // 当前行的上一行
		const uchar * current = src.ptr<const uchar>(i); //当前行
		const uchar * next = src.ptr<const uchar>(i + 1); //当前行的下一行
		for (int j = 1; j < src.cols-1; j++)
		{
			//选择P点
			//若该点P的像素值为0(即黑色)
			if ((int)current[j] == 0)
			{
				//若四邻域中至少有3个像素点为黑色,上下左右
				int domain[4];
				domain[0] = previous[j];
				domain[1] = next[j];
				domain[2] = current[j - 1];
				domain[3] = current[j + 1];

				if ((domain[0] == 0 && domain[1] == 0 && domain[2] == 0) || (domain[0] == 0 && domain[1] == 0 && domain[3] == 0)
					|| (domain[0] == 0 && domain[2] == 0 && domain[3] == 0) || (domain[1] == 0 && domain[2] == 0 && domain[3] == 0) ||
					(domain[0] == 0 && domain[1] == 0 && domain[2] == 0 && domain[3] == 0))
				{
					//则将点P视为文本上的像素，且将其置为蓝色
					dst.at<cv::Vec3b>(i, j)[0] = 255;
					dst.at<cv::Vec3b>(i, j)[1] = 0;
					dst.at<cv::Vec3b>(i, j)[2] = 0;
					temp.at<uchar>(i, j) = 1;

					if (domain[0] == 0)
					{
						//则将点P视为文本上的像素，且将其置为蓝色
						dst.at<cv::Vec3b>(i-1, j)[0] = 255;
						dst.at<cv::Vec3b>(i-1, j)[1] = 0;
						dst.at<cv::Vec3b>(i-1, j)[2] = 0;

						temp.at<uchar>(i-1, j) = 1;
					}
					if (domain[1] == 0)
					{
						//则将点P视为文本上的像素，且将其置为蓝色
						dst.at<cv::Vec3b>(i + 1, j)[0] = 255;
						dst.at<cv::Vec3b>(i + 1, j)[1] = 0;
						dst.at<cv::Vec3b>(i + 1, j)[2] = 0;

						temp.at<uchar>(i+1, j) = 1;

					}
					if (domain[2] == 0)
					{
						//则将点P视为文本上的像素，且将其置为蓝色
						dst.at<cv::Vec3b>(i, j-1)[0] = 255;
						dst.at<cv::Vec3b>(i, j-1)[1] = 0;
						dst.at<cv::Vec3b>(i, j-1)[2] = 0;

						temp.at<uchar>(i, j-1) = 1;

					}
					if (domain[3] == 0)
					{
						//则将点P视为文本上的像素，且将其置为蓝色
						dst.at<cv::Vec3b>(i, j+1)[0] = 255;
						dst.at<cv::Vec3b>(i, j+1)[1] = 0;
						dst.at<cv::Vec3b>(i, j+1)[2] = 0;

						temp.at<uchar>(i, j+1) = 1;
					}
				}
			}
		}
	}
}

//背景去除
void dislodgeBackGround(Mat& src, Mat& temp, Mat& dst1, Mat& dst2, int T)
{
	const int channels = src.channels();
	//基准线提取
	for (int i = 1; i < src.rows - 1; i++)
	{
		const uchar * previous = src.ptr<const uchar>(i - 1); // 当前行的上一行
		const uchar * current = src.ptr<const uchar>(i); //当前行
		const uchar * next = src.ptr<const uchar>(i + 1); //当前行的下一行
		for (int j = 1; j < src.cols - 1; j++)
		{
			//遍历整幅验证码图中的黑色像素，若该点不属于在3.3.1节中所提取的字符边界，将它设为点P
			if ((int)current[j] == 0 && (int)temp.at<uchar>(i, j) != 1)
			{
				int P[9];
				P[0] = current[j];
				P[1] = previous[j - 1];
				P[2] = previous[j];
				P[3] = previous[j + 1];
				P[4] = current[j + 1];
				P[5] = next[j + 1];
				P[6] = next[j];
				P[7] = next[j - 1];
				P[8] = current[j - 1];

				//点P2、P3、P6和P7，如果这四个位置都是白色
				if (P[2] == 255 && P[3] == 255 && P[6] == 255 && P[7] == 255)
				{
					//若P4和P5位置只有一个方向有黑色像素，P1和P8位置也只有一个方向有黑色像素
					if (((P[4] == 0 && P[5] != 0) || (P[4] != 0 && P[5] == 0)) &&
						((P[1] == 0 && P[8] != 0) || (P[1] != 0 && P[8] == 0)))
					{
						//则将点P视为基准线上的像素，且将其置为绿色
						dst1.at<Vec3b>(i, j)[0] = 0;
						dst1.at<Vec3b>(i, j)[1] = 255;
						dst1.at<Vec3b>(i, j)[2] = 0;

						temp.at<uchar>(i, j) = 2;
					}
				}
				else
				{
					//若P3、P5同时为黑色，或P1、P7同时为黑色
					if ((P[3] == 0 && P[5] == 0) || (P[1] == 0 && P[7] == 0))
					{
						//则将点P视为基准线上的像素，且将其置为绿色
						dst1.at<Vec3b>(i, j)[0] = 0;
						dst1.at<Vec3b>(i, j)[1] = 255;
						dst1.at<Vec3b>(i, j)[2] = 0;

						temp.at<uchar>(i, j) = 2;
					}
				}
			}
		}
	}

	dst1.copyTo(dst2);

	//背景填充
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			//若同一行中两个基准线上的点之间的距离D<T
			int D = 1000;
			if ((int)temp.at<uchar>(i, j) == 2)
			{
				int first=j;
				int second=0; 
				bool isBack = true;
				for (int k = j+1; k < (j + T) && k<src.cols; k++)
				{
					if ((int)temp.at<uchar>(i, k) == 2)
					{
						second = k;
						break;
					}
				}
				//且两点之间没有黑色像素或字符边界像素
				if (second<(j + T) && second<src.cols && second>first)
				{
					for (int k = first+1; k < second; k++)
					{
						//if (((int)dst1.at<Vec3b>(i, k)[0] == 0 && (int)dst1.at<Vec3b>(i, k)[1] == 0 && (int)dst1.at<Vec3b>(i, k)[2] == 0) 
						//	|| (int)temp.at<uchar>(i, k) == 1)
						if ((int)src.at<uchar>(i, k) == 0 && (int)temp.at<uchar>(i, k) == 1)
						{
							isBack = false;
							break;
						}
					}
				}
				else
				{
					isBack = false;
				}
				//将这两点之间的所有的像素都填充为图像背景,变为灰色
				if (isBack == true)
				{
					for (int k = first; k <= second; k++)
					{
						dst2.at<Vec3b>(i, k)[0] = 96;
						dst2.at<Vec3b>(i, k)[1] = 96;
						dst2.at<Vec3b>(i, k)[2] = 96;
					}
				}
			}
		}
	}
}

//去噪
void denoise(Mat& src, Mat &dst)
{
	morphologyEx(src, dst, MORPH_OPEN, NULL,Point(-1,-1),5);
}

//字符提取,二次提取
//templateSize是在二次判断字符是否为边界时使用的模板范围
//morphSize是形态学闭操作的核大小
//loDiff为漫水填充时，表示当前观察像素值与其部件邻域像素值或者待加入该部件的种子像素之间的亮度或颜色之负差（lower brightness/color difference）的最大值。  
//upDiff为漫水填充时，表示当前观察像素值与其部件邻域像素值或者待加入该部件的种子像素之间的亮度或颜色之正差（lower brightness/color difference）的最大值。
void extractCharacter(Mat& src, Mat& temp, Mat& dst1,Mat& dst2, int T1, int templateSize,int morphSize,int loDiff,int upDiff)
{
	src.copyTo(dst1);

	//漫水填充算法
	//填充边界
	Rect ccomp;
	//floodFill(dst, Point(0, 0), Scalar(96, 96, 96),&ccomp);
	//floodFill(dst, Point(dst.cols / 2, dst.rows / 2), Scalar(96, 96, 96), &ccomp);
	floodFill(dst1, Point(0, 0), Scalar(96, 96, 96), &ccomp, Scalar(loDiff, loDiff, loDiff), Scalar(upDiff, upDiff, upDiff));
	//floodFill(dst, Point(dst.cols/2, dst.rows/2), Scalar(96, 96, 96), &ccomp, Scalar(20, 20, 20), Scalar(20, 20, 20));


	//初步提取字符表面，作为去除字符边界噪声的依据
	//将将字符表面上的框架置为白色
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if ((int)dst1.at<Vec3b>(i, j)[0] == 0 && (int)dst1.at<Vec3b>(i, j)[1] == 0 && (int)dst1.at<Vec3b>(i, j)[2] == 0)
			{
				dst1.at<Vec3b>(i, j)[0] = 255;
				dst1.at<Vec3b>(i, j)[1] = 255;
				dst1.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}

	//水平方向遍历图像，若同一行中连续为白色像素的区域宽度 ，则先将该区域视为字符表面。
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (dst1.at<Vec3b>(i, j)[0] == 255 && dst1.at<Vec3b>(i, j)[1] == 255 && dst1.at<Vec3b>(i, j)[2] == 255)
			{
				int first = j;
				int second;
				bool isCharacter = true;
				for (int k = first + 1; k <= (first + T1) &&k<src.cols; k++)
				{
					if (dst1.at<Vec3b>(i, k)[0] != 255 || dst1.at<Vec3b>(i, k)[1] != 255 || dst1.at<Vec3b>(i, k)[2] != 255)
					{
						isCharacter = false;
						break;
					}
				}
				if (isCharacter)
				{
					for (int k = first; k <= (first + T1)&& k<=src.cols; k++)
					{
						temp.at<uchar>(i, k) = 3;
						
						/*
						dst.at<cv::Vec3b>(i, k)[0] = 0;
						dst.at<cv::Vec3b>(i, k)[1] = 0;
						dst.at<cv::Vec3b>(i, k)[2] = 255;
						*/
					}
				}
			}
		}
	}

	dst1.copyTo(dst2);
	//二次提取边界
	for (int i = templateSize; i < temp.rows - templateSize; i++)
	{
		for (int j = templateSize; j < temp.cols - templateSize; j++)
		{
			if ((int)temp.at<uchar>(i, j) == 1)
			{
				bool isCharacterEdge = false;
				//判断点A附近是否存在字符表面像素
				for (int n = i - templateSize; n < (i + templateSize); n++)
				{
					for (int m = j - templateSize; m < (j + templateSize); m++)
					{
						if ((int)temp.at<uchar>(n, m) == 3)
						{
							isCharacterEdge = true;
							break;
						}
					}
				}
				
				//如果存在，确认点A属于字符边界
				if (isCharacterEdge)
				{
					(int)temp.at<uchar>(i, j) == 1;
					dst1.at<cv::Vec3b>(i, j)[0] = 255;
					dst1.at<cv::Vec3b>(i, j)[1] = 0;
					dst1.at<cv::Vec3b>(i, j)[2] = 0;

					dst2.at<cv::Vec3b>(i, j)[0] = 255;
					dst2.at<cv::Vec3b>(i, j)[1] = 255;
					dst2.at<cv::Vec3b>(i, j)[2] = 255;
				}
				//否则点A属于图像背景。
				else
				{
					(int)temp.at<uchar>(i, j) == 0;
					dst1.at<cv::Vec3b>(i, j)[0] = 96;
					dst1.at<cv::Vec3b>(i, j)[1] = 96;
					dst1.at<cv::Vec3b>(i, j)[2] = 96;

					dst2.at<cv::Vec3b>(i, j)[0] = 96;
					dst2.at<cv::Vec3b>(i, j)[1] = 96;
					dst2.at<cv::Vec3b>(i, j)[2] = 96;
				}
			}
		}
	}
	//将图像都转为背景和字符
	for (int i = 0; i < temp.rows; i++)
	{
		for (int j = 0; j < temp.cols; j++)
		{
			if ((int)temp.at<uchar>(i, j) == 2 )
			{
				dst1.at<cv::Vec3b>(i, j)[0] = 96;
				dst1.at<cv::Vec3b>(i, j)[1] = 96;
				dst1.at<cv::Vec3b>(i, j)[2] = 96;

				dst2.at<cv::Vec3b>(i, j)[0] = 96;
				dst2.at<cv::Vec3b>(i, j)[1] = 96;
				dst2.at<cv::Vec3b>(i, j)[2] = 96;
			}
		}
	}

	//中值滤波
	medianBlur(dst2, dst2, 5);

	//形态学滤波,闭操作
	Mat element = getStructuringElement(MORPH_RECT, Size(morphSize, morphSize));
	//morphologyEx(dst2, dst2, MORPH_OPEN, element);//开运算
	morphologyEx(dst2, dst2, MORPH_CLOSE, element);//闭运算
	//morphologyEx(dst2, dst2, MORPH_GRADIENT, element);//形态学梯度
	//morphologyEx(dst2, dst2, MORPH_ERODE, element);//腐蚀
	//morphologyEx(dst2, dst2, MORPH_DILATE, element);//膨胀

	//二值化
	threshold(dst2, dst2, 128, 255, cv::THRESH_BINARY);
}



int main(int argc, char** argv)
{
	Mat image = imread(".././data/~E5]P7]DGBB$OE6(A(5D3VC.png ");//[D3B1L@W31W7U)BJQI$0SCL.png  `QYGD2HE)[P}LS1FQ~)8M1F.png  }`696BF9SVM`HK$1THWA[KK.png  %N@GM8%XY7DTDY]YQ1`}U6A.png {71ZW78PSEXN}T(04OIMAGK.png  D)`}O9B4L07E`(59(PL35]H.png 
	Mat gray;
	Mat binImage;
	Mat temp(image.size(),CV_8U);
	Mat FirstImage,SecondImage,ThirdImage,FourthImgae,FifthImage;

	int T = 8, T1 = 6, templateSize = 5, morphSize=3,loDiff=220,upDiff=140;

	if(image.empty())
	{
		cout<<"image empty!"<<endl;
		return 0;
	}

	//初始化设置矩阵
	initTemp(temp);

	//灰度化
	cvtColor(image, gray, CV_BGR2GRAY);
	//二值化
	threshold(gray, binImage, 128, 255, cv::THRESH_BINARY);

	//选择初步文本像素
	////提取字符边界
	image.copyTo(FirstImage);
	selectTextFirst(binImage, FirstImage,temp);

	//背景去除
	FirstImage.copyTo(SecondImage);
	dislodgeBackGround(binImage, temp, SecondImage, ThirdImage,T);

	//字符提取
	extractCharacter(ThirdImage, temp, FourthImgae, FifthImage, T1, templateSize, morphSize, loDiff, upDiff);

	
	//显示图像
	imshow("样本原图", image);
	imshow("字符边界初步结果", FirstImage);
	imshow("基准线提取", SecondImage);
	imshow("背景填充", ThirdImage);
	imshow("二次提取字符边界", FourthImgae);
	imshow("字符提取初步结果", FifthImage);
	waitKey(0);


	return 0;
}