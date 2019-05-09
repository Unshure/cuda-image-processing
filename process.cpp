#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;

Mat grayscale(Mat image) {
    Mat grayImage(image.rows, image.cols, CV_8UC1);
    for(int x = 0; x < grayImage.cols; x++) {
        for (int y = 0; y < grayImage.rows; y++) {
            Vec3b intensity = image.at<Vec3b>(y, x);
            grayImage.at<uchar>(y, x) = (.3*intensity[2]) + (.59 * intensity[1]) + (.11 * intensity[0]);
        }
    }
    return grayImage;
}

Mat blur(Mat image) {
    Mat blurImage(image);
    /*for(int x = 0; x < grayImage.cols; x++) {
        for (int y = 0; y < grayImage.rows; y++) {
            Vec3b intensity = image.at<Vec3b>(y, x);
            grayImage.at<uchar>(y, x) = (.3*intensity[2]) + (.59 * intensity[1]) + (.11 * intensity[0]);
        }
    }*/
    return blurImage;
}

int main(int argc, char** argv )
{
    if ( argc != 2 )
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
    
    Mat grayImage = blur(image);

    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", grayImage);
    waitKey(0);

    return 0;
}