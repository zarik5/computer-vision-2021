#include <iostream>
#include <opencv2/opencv.hpp>

#include "denoise_tests.h"
#include "histogram_tests.h"
#include "morphology_tests.h"

const char *const INPUT_IMAGE_FNAME = "overexposed.jpg";

int main() {
    auto input = cv::imread(std::string("../../../data/") + INPUT_IMAGE_FNAME);
    cv::imshow("Original image", input);
    cv::waitKey();

    // Histogram tests
    manipulate_histograms_rgb(input);
    auto equalized_image = manipulate_histograms_hsv(input);

    // Denoise tests

    auto median_params = test_median_filter(equalized_image);
    std::cout << "Selected median filter parameters: " << std::endl
              << "Kernel size: " << median_params.kernel_size << std::endl
              << std::endl;

    auto gaussian_params = test_gaussian_filter(equalized_image);
    std::cout << "Selected gaussian filter parameters: " << std::endl
              << "Kernel size: " << gaussian_params.kernel_size << std::endl
              << "Sigma: " << gaussian_params.sigma << std::endl
              << std::endl;

    auto bilateral_params = test_bilateral_filter(equalized_image);
    std::cout << "Selected gaussian filter parameters: " << std::endl
              << "Kernel size: " << bilateral_params.kernel_size << std::endl
              << "Sigma range: " << bilateral_params.sigma_range << std::endl
              << "Sigma space: " << bilateral_params.sigma_space << std::endl
              << std::endl;

    // Morphological tests
    morphology_tests(equalized_image);
}
