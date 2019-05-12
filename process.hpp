#ifndef PROCESS_H
#define PROCESS_H
#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
uchar* grayscale(uchar* image, int rows, int cols, int channels, int step);

int* kernelSum(uchar* image, int rows, int cols, int channels, int step, int x, int y, int size);

uchar* blur(uchar* image, int rows, int cols, int channels, int step, int size);

int kernelLineDetect(uchar* image, int rows, int cols, int x, int y);

uchar* detectLine(uchar* image, int rows, int cols, int channels, int step);

int kernelArray[4][3][3] = {{{-1,-1,-1},{2,2,2},{-1,-1,-1}},
                            {{-1,2,-1},{-1,2,-1},{-1,2,-1}},
                            {{-1,-1,2},{-1,2,-1},{2,-1,-1}},
                            {{2,-1,-1},{-1,2,-1},{-1,-1,2}}};
#endif
