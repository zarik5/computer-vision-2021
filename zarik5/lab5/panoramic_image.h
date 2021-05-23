#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// Note: The functionality of this class could have been implemented in a simple static function
class PanoramicImage {
  public:
    PanoramicImage(std::vector<cv::Mat> input_images);
    cv::Mat PanoramicImage::stitch(float half_fov_deg, float distance_ratio);

  private:
    std::vector<cv::Mat> input_images;
};