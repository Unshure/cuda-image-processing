#include "process.hpp"

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread( argv[1], 1);

    Mat *cudaImage;
    cudaMallocManaged(&cudaImage, image.total() * image.elemSize());
    memcpy(cudaImage, &image, image.total() * image.elemSize());
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    
    grayscale(cudaImage);

    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", blurImage);
    waitKey(0);


    cudaFree(cudaImage);
    return 0;
}

__global__
void grayscale(Mat *cudaImage) {
    Mat image = *cudaImage;

    int index = threadIdx.x;
    int stride = blockDim.x;

    int numPixels = image.rows * image.cols;

    Mat grayImage(image.rows, image.cols, CV_8UC1);

    for (int i = index; i < numPixels; i+= stride) {
        int y = index / image.cols;
        int x = index % image.cols;
        Vec3b intensity = image.at<Vec3b>(y,x);
        grayImage.at<uchar>(y,x) = (.3*intensity[2]) + (.59 * intensity[1]) + (.11 * intensity[0]);
    }
    grayImage.copyTo(image);

    /*
    for(int x = 0; x < grayImage.cols; x++) {
        for (int y = 0; y < grayImage.rows; y++) {
            Vec3b intensity = image.at<Vec3b>(y, x);
            grayImage.at<uchar>(y, x) = (.3*intensity[2]) + (.59 * intensity[1]) + (.11 * intensity[0]);
        }
    }
    return grayImage;
    */
}

__device__
int* kernelSum(Mat image, int x, int y, int size) {

    int *sum = (int*)malloc(3* sizeof(int));
    memset(sum, 0, sizeof(int)*3)

    int numPixels = 0;
    for (int i = (x - (size/2)); i < (x + (size/2))+1; i++) {
        for (int j = (y - (size/2)); j < (y + (size/2))+1; j++) {
            if (i >= 0 && j >= 0 && i < image.cols && j < image.rows) {
                Vec3b intensity = image.at<Vec3b>(j, i);
                sum[0] += intensity[0];
                sum[1] += intensity[1];
                sum[2] += intensity[2];
                numPixels++;
            }
        }
    }
    sum[0] = sum[0] / numPixels;
    sum[1] = sum[1] / numPixels;
    sum[2] = sum[2] / numPixels;
    return sum;
}

__global__
void blur(Mat *cudaImage, int size) {

    Mat image = *cudaImage;

    int index = threadIdx.x;
    int stride = blockDim.x;

    int numPixels = image.rows * image.cols;

    Mat blurImage = image.clone();

    for (int i = index; i < numPixels; i += stride) {
        int y = index / image.cols;
        int x = index % image.cols;

        int *average = kernelSum(image, x, y, size);
        blurImage.at<Vec3b>(y, x)[0] = average[0];
        blurImage.at<Vec3b>(y, x)[1] = average[1];
        blurImage.at<Vec3b>(y, x)[2] = average[2];
    }
    blurImage.copyTo(image);

    /*
    for(int x = 0; x < image.cols; x++) {
        for (int y = 0; y < image.rows; y++) {
            int *average = kernelSum(image, x, y, size);
            blurImage.at<Vec3b>(y, x)[0] = average[0];
            blurImage.at<Vec3b>(y, x)[1] = average[1];
            blurImage.at<Vec3b>(y, x)[2] = average[2];
        } 
    }
    return blurImage;
    */
}

__device__
int kernelLineDetect(Mat image, int x, int y) {
    int sum = 0;
    int numPixels = 0;
    int kx = 0;
    for (int i = (x - 1); i < (x + 2); i++) {
        int ky = 0;
        for (int j = (y - 1); j < (y + 2); j++) {
            if (i >= 0 && j >= 0 && i < image.cols && j < image.rows) {
                for(int k = 0; k < 4; k ++) {
                    sum += kernelArray[k][kx][ky] * image.at<uchar>(j, i);
                }
                numPixels++;
            }
            ky++;
        }
        kx++;
    }
    
    return sum / (numPixels*4);
}

__global__
Mat detectLine(Mat *cudaImage) {
    //Assuming gray image input

    Mat image = *cudaImage;

    int index = threadIdx.x;
    int stride = blockDim.x;

    int numPixels = image.rows * image.cols;

    Mat lineImage = image.clone();

    for(int i = index; i < numPixels; i+= stride) {
        int y = index / image.cols;
        int x = index % image.cols;

        lineImage.at<uchar>(y, x) = kernelLineDetect(image, x, y);
    }
    lineImage.copyTo(image);

    /*
    for(int x = 0; x < image.cols; x++) {
        for (int y = 0; y < image.rows; y++) {
            lineImage.at<uchar>(y, x) = kernelLineDetect(grayImage, x, y);
        } 
    }
    return lineImage;
    */
}