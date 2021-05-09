#pragma once

#include "filter.h"
#include <opencv2/opencv.hpp>

MedianFilterParams test_median_filter(cv::Mat image);

GaussianFilterParams test_gaussian_filter(cv::Mat image);

BilateralFilterParams test_bilateral_filter(cv::Mat image);