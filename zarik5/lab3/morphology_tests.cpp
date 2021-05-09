#include "morphology_tests.h"

void morphology_tests(cv::Mat image) {
    cv::imshow("Original", image);
    cv::waitKey();

    cv::Mat structuring_element = cv::Mat::zeros(5, 5, CV_8UC1);

    // vertical bar
    structuring_element.at<uchar>(2, 0) = 255;
    structuring_element.at<uchar>(2, 1) = 255;
    structuring_element.at<uchar>(2, 2) = 255;
    structuring_element.at<uchar>(2, 3) = 255;
    structuring_element.at<uchar>(2, 4) = 255;

    // corners
    structuring_element.at<uchar>(0, 0) = 255;
    structuring_element.at<uchar>(0, 4) = 255;
    structuring_element.at<uchar>(4, 4) = 255;
    structuring_element.at<uchar>(4, 0) = 255;

    cv::Mat filtered_image;
    cv::morphologyEx(image, filtered_image, cv::MORPH_CLOSE, structuring_element);
    cv::imshow("Open filter", filtered_image);
    cv::waitKey();
}