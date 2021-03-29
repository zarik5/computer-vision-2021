#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("../data/robocup.jpg");
    imshow("Display Image", image);
    cv::waitKey(0);

    std::cout << "Hello" << std::endl;
    std::cout << "World" << std::endl;

    
}