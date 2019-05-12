#ifndef CUDAPROCESS_H
#define CUDAPROCESS_H

uchar* cudaGrayscale(uchar* image, int rows, int cols, int channels, int step);

uchar* cudaBlur(uchar* image, int rows, int cols, int channels, int step, int size);

uchar* cudaDetectLine(uchar* image, int rows, int cols, int channels, int step);

#endif