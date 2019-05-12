#include "cudaProcess.h"

__global__
void execCudaGrayscale(unsigned char* image, unsigned char* grayImage, int rows, int cols, int channels, int step) {

    int index = threadIdx.x;
    int stride = blockDim.x;

    int numPixels = rows * cols;

    for (int i = index; i < numPixels; i+= stride) {
        int y = index / cols;
        int x = index % cols;

        int blue = (int)image[channels*x + step*y];
        int green = (int)image[channels*x + step*y + 1];
        int red = (int)image[channels*x + step*y + 2];
        
        grayImage[x + cols*y] = (unsigned char)(.3*red) + (.59 * green) + (.11 * blue);
    }
}

__device__
void cudaKernelSum(unsigned char* image, int rows, int cols, int channels, int step, int x, int y, int size, int* sum) {
    int numPixels = 0;
    for (int i = (x - (size/2)); i < (x + (size/2))+1; i++) {
        for (int j = (y - (size/2)); j < (y + (size/2))+1; j++) {
            if (i >= 0 && j >= 0 && i < cols && j < rows) {
                sum[0] += image[i*channels + y*step];
                sum[1] += image[i*channels + y*step + 1];
                sum[2] += image[i*channels + y*step + 2];
                numPixels++;
            }
        }
    }
    sum[0] = sum[0] / numPixels;
    sum[1] = sum[1] / numPixels;
    sum[2] = sum[2] / numPixels;
}

__global__
void execCudaBlur(unsigned char* image, unsigned char* blurImage, int rows, int cols, int channels, int step, int size) {

    int index = threadIdx.x;
    int stride = blockDim.x;

    int numPixels = rows * cols;

    int *sum = (int*)malloc(3 * sizeof(int));
    memset(sum, 0, 3*sizeof(int));

    for (int i = index; i < numPixels; i += stride) {
        int y = index / cols;
        int x = index % cols;

        cudaKernelSum(image, rows, cols, channels, step, x, y, size, sum);
        blurImage[channels*x + step*y] =     sum[0];
        blurImage[channels*x + step*y + 1] = sum[1];
        blurImage[channels*x + step*y + 2] = sum[2];
    }
}

__device__
void cudaKernelLineDetect(unsigned char* image, int rows, int cols, int x, int y, int* val) {
    int sum = 0;
    int numPixels = 0;
    int kx = 0;
    int cudaKernelArray[4][3][3] = {{{-1,-1,-1},{2,2,2},{-1,-1,-1}},
                            {{-1,2,-1},{-1,2,-1},{-1,2,-1}},
                            {{-1,-1,2},{-1,2,-1},{2,-1,-1}},
                            {{2,-1,-1},{-1,2,-1},{-1,-1,2}}};
    for (int i = (x - 1); i < (x + 2); i++) {
        int ky = 0;
        for (int j = (y - 1); j < (y + 2); j++) {
            if (i >= 0 && j >= 0 && i < cols && j < rows) {
                for(int k = 0; k < 4; k ++) {
                    sum += cudaKernelArray[k][kx][ky] * image[i + cols*j];
                }
                numPixels++;
            }
            ky++;
        }
        kx++;
    }
    *val = sum / (numPixels*4);
}

__global__
void execCudaDetectLine(unsigned char* image, unsigned char* lineImage, int rows, int cols, int channels, int step) {
    //Assuming gray image input

    int index = threadIdx.x;
    int stride = blockDim.x;

    int numPixels = rows * cols;
    int *val;

    for(int i = index; i < numPixels; i+= stride) {
        int y = index / cols;
        int x = index % cols;
        cudaKernelLineDetect(image, rows, cols, x, y, val);
        lineImage[x + cols*y] = *val;
    }
}

unsigned char* cudaGrayscale(unsigned char* image, int rows, int cols, int channels, int step) {

    int threadsPerBlock = 1024;
    int numBlocks = 65000;

    unsigned char* cudaImage;
    unsigned char* cudaGrayImage;
    cudaMallocManaged(&cudaImage, sizeof(unsigned char)*rows*cols);
    cudaMallocManaged(&cudaGrayImage, sizeof(unsigned char)*rows*cols*channels);
    
    cudaMemcpy(cudaImage, image, sizeof(unsigned char)*rows*cols*channels, cudaMemcpyHostToDevice);
    memset(cudaGrayImage, 0, sizeof(unsigned char)*rows*cols);

    execCudaGrayscale<<<numBlocks, threadsPerBlock>>>(cudaImage, cudaGrayImage, rows, cols, channels, step);
    cudaDeviceSynchronize();

    unsigned char* grayImage = (unsigned char*)malloc(sizeof(unsigned char)*rows*cols);
    cudaMemcpy(grayImage, cudaGrayImage, sizeof(unsigned char)*rows*cols, cudaMemcpyDeviceToHost);

    cudaFree(cudaImage);
    cudaFree(cudaGrayImage);
    return grayImage;

}

unsigned char* cudaBlur(unsigned char* image, int rows, int cols, int channels, int step, int size) {

    int threadsPerBlock = 1024;
    int numBlocks = 65000;

    unsigned char* cudaImage;
    unsigned char* cudaBlurImage;
    cudaMallocManaged(&cudaImage, sizeof(unsigned char)*rows*cols*channels);
    cudaMallocManaged(&cudaBlurImage, sizeof(unsigned char)*rows*cols*channels);

    cudaMemcpy(cudaImage, image, sizeof(unsigned char)*rows*cols*channels, cudaMemcpyHostToDevice);
    memset(cudaBlurImage, 0, sizeof(unsigned char)*rows*cols*channels);

    execCudaBlur<<<numBlocks, threadsPerBlock>>>(cudaImage, cudaBlurImage, rows, cols, channels, step, size);
    cudaDeviceSynchronize();

    unsigned char* blurImage = (unsigned char*)malloc(sizeof(unsigned char)*rows*cols*channels);
    cudaMemcpy(blurImage, cudaBlurImage, sizeof(unsigned char)*rows*cols*channels, cudaMemcpyDeviceToHost);

    cudaFree(cudaImage);
    cudaFree(cudaBlurImage);
    return blurImage;

}

unsigned char* cudaDetectLine(unsigned char* image, int rows, int cols, int channels, int step) {

    int threadsPerBlock = 1024;
    int numBlocks = 65000;

    unsigned char* grayImage = cudaGrayscale(image, rows, cols, channels, step);

    unsigned char* cudaImage;
    unsigned char* cudaLineImage;
    cudaMallocManaged(&cudaImage, sizeof(unsigned char)*rows*cols);
    cudaMallocManaged(&cudaLineImage, sizeof(unsigned char)*rows*cols);

    cudaMemcpy(cudaImage, grayImage, sizeof(unsigned char)*rows*cols, cudaMemcpyHostToDevice);
    memset(cudaLineImage, 0, sizeof(unsigned char)*rows*cols);

    execCudaDetectLine<<<numBlocks, threadsPerBlock>>>(cudaImage, cudaLineImage, rows, cols, channels, step);
    cudaDeviceSynchronize();

    unsigned char* lineImage = (unsigned char*)malloc(sizeof(unsigned char)*rows*cols);
    cudaMemcpy(lineImage, cudaLineImage, sizeof(unsigned char)*rows*cols, cudaMemcpyDeviceToHost);

    cudaFree(cudaImage);
    cudaFree(cudaLineImage);
    return lineImage;

}
