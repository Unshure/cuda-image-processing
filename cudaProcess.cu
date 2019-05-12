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
        
        grayImage[x + cols*y] = (uchar)(.3*red) + (.59 * green) + (.11 * blue);
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
void execCudaBlur(unsigned char* image, unsigned char* blurImage, int rows, int cols, int channels, int step int size) {

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
void cudaKernelLineDetect(unsigned char image, int rows, int cols, int x, int y, int* val) {
    int sum = 0;
    int numPixels = 0;
    int kx = 0;
    for (int i = (x - 1); i < (x + 2); i++) {
        int ky = 0;
        for (int j = (y - 1); j < (y + 2); j++) {
            if (i >= 0 && j >= 0 && i < cols && j < rows) {
                for(int k = 0; k < 4; k ++) {
                    sum += kernelArray[k][kx][ky] * image[i + cols*j];
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
Mat execCudaDetectLine(unsigned char* image, unsigned char* lineImage, int rows, int cols, int channels, int step) {
    //Assuming gray image input

    int index = threadIdx.x;
    int stride = blockDim.x;

    int numPixels = image.rows * image.cols;

    Mat lineImage = image.clone();

    for(int i = index; i < numPixels; i+= stride) {
        int y = index / image.cols;
        int x = index % image.cols;

        lineImage[x + cols*y] = cudaKernelLineDetect(image, rows, cols, x, y);
    }
}

uchar* cudaGrayscale(unsigned char* image, int rows, int cols, int channels, int step) {
    
    uchar* cudaImage;
    uchar* cudaGrayImage;
    cudaMallocManaged(cudaImage, sizeof(uchar)*rows*cols);
    cudaMallocManaged(cudaGrayImage, sizeof(uchar)*rows*cols*channels);
    
    memcpy(cudaImage, image, sizeof(uchar)*rows*cols*channels);
    memset(cudaGrayImage, 0, sizeof(uchar)*rows*cols);

    execCudaGrayscale<<<numBlocks, threadsPerBlock>>>(cudaImage, cudaGrayImage, rows, cols, channels, step);
    cudaDeviceSynchronize();

    uchar* grayImage = (uchar*)malloc(sizeof(uchar)*rows*cols);
    memcpy(grayImage, cudaGrayImage, sizeof(uchar)*rows*cols);

    cudaFree(cudaImage);
    cudaFree(cudaGrayImage);
    return grayImage;

}

uchar* cudaBlur(uchar* image, int rows, int cols, int channels, int step, int size) {

    uchar* cudaImage;
    uchar* cudaBlurImage;
    cudaMallocManaged(cudaImage, sizeof(uchar)*rows*cols*channels);
    cudaMallocManaged(cudaBlurImage, sizeof(uchar)*rows*cols*channels);

    memcpy(cudaImage, image, sizeof(uchar)*rows*cols*channels);
    memset(cudaBlurImage, 0, sizeof(uchar)*rows*cols*channels);

    execCudaBlur<<<numBlocks, threadsPerBlock>>>(cudaImage, cudaBlurImage, rows, cols, channels, step, size);
    cudaDeviceSynchronize();

    uchar* blurImage = (uchar*)malloc(sizeof(uchar)*rows*cols*channels);
    memcpy(blurImage, cudaBlurImage, sizeof(uchar)*rows*cols*channels);

    cudaFree(cudaImage);
    cudaFree(cudaBlurImage);
    return blurImage;

}

uchar* cudaDetectLine(uchar* image, int rows, int cols, int channels, int step) {

    uchar* grayImage = cudaGrayscale(image, rows, cols, channels, step);

    uchar* cudaImage;
    uchar* cudaLineImage;
    cudaMallocManaged(cudaImage, sizeof(uchar)*rows*cols);
    cudaMallocManaged(cudaLineImage, sizeof(uchar)*rows*cols);

    memcpy(cudaImage, grayImage, sizeof(uchar)*rows*cols)
    memset(cudaLineImage, 0, sizeof(uchar)*rows*cols);

    execCudaDetectLine<<<numBlocks, threadsPerBlock>>>(cudaImage, cudaLineImage, rows, cols, channels, step, size);
    cudaDeviceSynchronize();

    uchar* lineImage = (uchar*)malloc(sizeof(uchar)*rows*cols);
    memcpy(lineImage, cudaLineImage, sizeof(uchar)*rows*cols);

    cudaFree(cudaImage);
    cudaFree(cudaLineImage);
    return lineImage;

}
