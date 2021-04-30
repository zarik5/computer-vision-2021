#pragma once

#include <opencv2/opencv.hpp>

struct LinesResult {
    cv::Point line1_p1;
    cv::Point line1_p2;
    cv::Point line2_p1;
    cv::Point line2_p2;
    int canny_threshold1;
    int canny_threshold2;
    float hough_rho_resolution;
    float hough_theta_resolution;
};

LinesResult get_2_strongest_lines_interactive(cv::Mat input);