#pragma once

#include <opencv2/opencv.hpp>

// Find edges in an image using Canny edge detector.
inline cv::Mat detect_edges(cv::Mat input, int threshold1, int threshold2) {
    cv::Mat output_edges;
    cv::Canny(input, output_edges, (double)threshold1, (double)threshold2, 3);
    return output_edges;
}