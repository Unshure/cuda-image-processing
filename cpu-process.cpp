#include "process.hpp"
//#include "cudaProcess.h"

#include <ctime>

int main(int argc, char** argv )
{
    if ( argc < 3 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    Mat processedImage;

    struct timespec start, finish;
    double elapsed;
    printf("-Starting Image Processor-\n");
    // Start Timer
    clock_gettime(CLOCK_MONOTONIC, &start);

    if (strcmp(argv[2], "-l") == 0) {

        unsigned char* lineImageData = detectLine(image.data, image.rows, image.cols, image.channels(), image.step);
        processedImage = Mat(image.rows, image.cols, CV_8UC1, lineImageData);
        
    }   else if (strcmp(argv[2], "-g") == 0) {
        unsigned char* grayImageData = grayscale(image.data, image.rows, image.cols, image.channels(), image.step);
        processedImage = Mat(image.rows, image.cols, CV_8UC1, grayImageData);
        
    }   else if (strcmp(argv[2], "-b") == 0 && argc == 4 && atoi(argv[3]) >= 0) {
        unsigned char* blurImageData = blur(image.data, image.rows, image.cols, image.channels(), image.step, atoi(argv[3]));
        processedImage = Mat(image.rows, image.cols, CV_8UC3, blurImageData);

    /*}   else if (strcmp(argv[2], "-cuda") == 0) {
            if (strcmp(argv[3], "-l") == 0) {
                unsigned char* lineImageData = cudaDetectLine(image.data, image.rows, image.cols, image.channels(), image.step);
                processedImage = Mat(image.rows, image.cols, CV_8UC1, lineImageData);

            }   else if (strcmp(argv[3], "-g") == 0) {
                unsigned char* grayImageData = cudaGrayscale(image.data, image.rows, image.cols, image.channels(), image.step);
                processedImage = Mat(image.rows, image.cols, CV_8UC1, grayImageData);

                
            }   else if (strcmp(argv[3], "-b") == 0 && argc == 5 && atoi(argv[4]) >= 0) {
                unsigned char* blurImageData = cudaBlur(image.data, image.rows, image.cols, image.channels(), image.step, atoi(argv[4]));
                processedImage = Mat(image.rows, image.cols, CV_8UC3, blurImageData);

            } else {
                printf("Missing cuda tag\n");
                return 0;
            }*/
    }   else {
        printf("Missing or wrong tag\n");
        return 0;
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Time: %f\n", elapsed);
        
    namedWindow("Display Image", WINDOW_AUTOSIZE );

    imshow("Display Image", processedImage);
    waitKey(0);

    return 0;
}


uchar* grayscale(uchar* image, int rows, int cols, int channels, int step) {
    
    uchar* grayImage = (uchar*)malloc(sizeof(uchar)*rows*cols);
    memset(grayImage, 0, sizeof(uchar)*rows*cols);
        
    for(int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int blue = (int)image[channels*x + step*y];
            int green = (int)image[channels*x + step*y + 1];
            int red = (int)image[channels*x + step*y + 2];
            
            grayImage[x + cols*y] = (uchar)(.3*red) + (.59 * green) + (.11 * blue);

        }
    }
    return grayImage;
}


int* kernelSum(uchar* image, int rows, int cols, int channels, int step, int x, int y, int size) {
    int *sum = (int*)malloc(3 * sizeof(int));
    memset(sum, 0, 3*sizeof(int));
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
    return sum;
}

uchar* blur(uchar* image, int rows, int cols, int channels, int step, int size) {

    uchar* blurImage = (uchar*)malloc(sizeof(uchar)*rows*cols*channels);
    memset(blurImage, 0, sizeof(uchar)*rows*cols*channels);
    
        
    for(int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int *average = kernelSum(image, rows, cols, channels, step, x, y, size);
            blurImage[channels*x + step*y] =     average[0];
            blurImage[channels*x + step*y + 1] = average[1];
            blurImage[channels*x + step*y + 2] = average[2];
        }
    }
    return blurImage;
}


int kernelLineDetect(uchar* image, int rows, int cols, int x, int y) {
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
    
    return sum / (numPixels*4);
}

uchar* detectLine(uchar* image, int rows, int cols, int channels, int step) {

    uchar* lineImage = (uchar*)malloc(sizeof(uchar)*rows*cols);
    memset(lineImage, 0, sizeof(uchar)*rows*cols);
    
    uchar* grayImage = grayscale(image, rows, cols, channels, step);
        
    for(int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {    
            lineImage[x + cols*y] = kernelLineDetect(grayImage, rows, cols, x, y);
        }
    }
    return lineImage;
}

