#include <iostream>

#include <opencv2/opencv.hpp>

void mouse_callback(int event, int x, int y, int flag, void *mouse_position_vptr) {
    auto *mouse_position_ptr = (cv::Point2d *)mouse_position_vptr;

    if (event == cv::EVENT_LBUTTONDOWN) {
        mouse_position_ptr->x = x;
        mouse_position_ptr->y = y;
    }
}

int main() {
    std::cout << "hello world" << std::endl;

    cv::Mat img = cv::imread("../../../data/robocup.jpg");

    cv::namedWindow("example");

    cv::Point2d mouse_position;

    cv::setMouseCallback("example", mouse_callback, (void *)&mouse_position);

    cv::imshow("example", img);

    int k = cv::waitKey(0);

    std::cout << mouse_position << std::endl;

    auto rect = cv::Rect(mouse_position.x - 4, mouse_position.y - 4, 9, 9);
    auto mean = cv::mean(img(rect));

    std::cout << mean << std::endl;
}
