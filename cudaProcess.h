#ifndef CUDAPROCESS_H
#define CUDAPROCESS_H

unsigned char* cudaGrayscale(unsigned char*, int, int, int, int);

unsigned char* cudaBlur(unsigned char*, int, int, int, int, int);

unsigned char* cudaDetectLine(unsigned char*, int, int, int, int);

#endif