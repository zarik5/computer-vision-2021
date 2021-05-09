#pragma once

#include <opencv2/opencv.hpp>

void manipulate_histograms_rgb(cv::Mat input_bgr);

cv::Mat manipulate_histograms_hsv(cv::Mat input_bgr);