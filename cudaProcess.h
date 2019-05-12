#ifndef CUDAPROCESS_H
#define CUDAPROCESS_H

unsigned char* cudaGrayscale(unsigned char*, int, int, int, int);

unsigned char* cudaBlur(unsigned char*, int, int, int, int, int);

unsigned char* cudaDetectLine(unsigned char*, int, int, int, int);

__constant__ int cudaKernelArray[4][3][3] = {{{-1,-1,-1},{2,2,2},{-1,-1,-1}},
                            {{-1,2,-1},{-1,2,-1},{-1,2,-1}},
                            {{-1,-1,2},{-1,2,-1},{2,-1,-1}},
                            {{2,-1,-1},{-1,2,-1},{-1,-1,2}}};

int threadsPerBlock = 1024;
int numBlocks = 65000;


#endif