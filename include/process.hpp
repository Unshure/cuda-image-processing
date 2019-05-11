#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;

Mat grayscale(Mat image);

int* kernelSum(Mat image, int x, int y, int size);

Mat blur(Mat image, int size);

int kernelLineDetect(Mat image, int x, int y);

Mat detectLine(Mat image);

int kernelArray[4][3][3] = {{{-1,-1,-1},{2,2,2},{-1,-1,-1}},
                            {{-1,2,-1},{-1,2,-1},{-1,2,-1}},
                            {{-1,-1,2},{-1,2,-1},{2,-1,-1}},
                            {{2,-1,-1},{-1,2,-1},{-1,-1,2}}};