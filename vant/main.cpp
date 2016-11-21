#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <rgb_hist.h>

using namespace std;
using namespace cv;


int main(int argc, char *argv[])
{
    Mat src, histImage;// dst;


    src = imread( argv[1], 1 );

    if( !src.data )
    {
        return -1;
    }

    //call rgb histogram
    histImage = rgb_hist(src);

    /// Display Hist RGB
    namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
    imshow("calcHist Demo", histImage );

    waitKey(0);

    return 0;
}
