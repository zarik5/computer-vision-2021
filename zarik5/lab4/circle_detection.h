#pragma once

#include <opencv2/opencv.hpp>

struct CircleResult {
    cv::Point circle_center;
    float circle_radius;
    int canny_threshold1;
    int canny_threshold2;
    float hough_resolution;
};

CircleResult get_strongest_circle_interactive(cv::Mat input);