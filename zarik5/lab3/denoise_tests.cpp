#include "denoise_tests.h"

const int INITIAL_KERNEL_SIZE = 3;
const int INITIAL_SIGMA = 3;
const int INITIAL_SIGMA_RANGE = 5;
const int INITIAL_SIGMA_SPACE = 5;

const int MAX_KERNEL_SIZE = 20;
const int MAX_SIGMA = 20;
const int MAX_SIGMA_RANGE = 256;
const int MAX_SIGMA_SPACE = 20;

struct OnTrackbarChangeData {
    // Filter is abstract and must be passed as a pointer
    Filter *filter;
    std::string window_name;
};

static void on_trackbar_change(int _, void *data_vptr) {
    auto *data_ptr = (OnTrackbarChangeData *)data_vptr;

    data_ptr->filter->do_filter();

    cv::imshow(data_ptr->window_name, data_ptr->filter->get_result());
}

MedianFilterParams test_median_filter(cv::Mat image) {
    const char *const WINDOW_NAME = "Denoised image with median filter";

    auto filter = MedianFilter(image, MedianFilterParams{INITIAL_KERNEL_SIZE});
    auto data = OnTrackbarChangeData{&filter, WINDOW_NAME};

    cv::namedWindow(WINDOW_NAME);
    cv::createTrackbar("Kernel size", WINDOW_NAME, filter.kernel_size_ptr_mut(), MAX_KERNEL_SIZE,
                       on_trackbar_change, &data);

    // Initialize view
    cv::imshow(WINDOW_NAME, filter.get_result());
    cv::waitKey();

    return filter.get_params();
}

GaussianFilterParams test_gaussian_filter(cv::Mat image) {
    const char *const WINDOW_NAME = "Denoised image with gaussian filter";

    auto filter = GaussianFilter(image, GaussianFilterParams{INITIAL_KERNEL_SIZE, INITIAL_SIGMA});
    auto data = OnTrackbarChangeData{&filter, WINDOW_NAME};

    cv::namedWindow(WINDOW_NAME);
    cv::createTrackbar("Kernel size", WINDOW_NAME, filter.kernel_size_ptr_mut(), MAX_KERNEL_SIZE,
                       on_trackbar_change, &data);
    cv::createTrackbar("Sigma", WINDOW_NAME, filter.sigma_ptr_mut(), MAX_SIGMA, on_trackbar_change,
                       &data);

    cv::imshow(WINDOW_NAME, filter.get_result());
    cv::waitKey();

    return filter.get_params();
}

BilateralFilterParams test_bilateral_filter(cv::Mat image) {
    const char *const WINDOW_NAME = "Denoised image with bilateral filter";

    auto filter =
        BilateralFilter(image, BilateralFilterParams{0, INITIAL_SIGMA_RANGE, INITIAL_SIGMA_SPACE});
    auto data = OnTrackbarChangeData{&filter, WINDOW_NAME};

    cv::namedWindow(WINDOW_NAME);
    cv::createTrackbar("Sigma range", WINDOW_NAME, filter.sigma_range_ptr_mut(), MAX_SIGMA_RANGE,
                       on_trackbar_change, &data);
    cv::createTrackbar("Sigma space", WINDOW_NAME, filter.sigma_space_ptr_mut(), MAX_SIGMA_SPACE,
                       on_trackbar_change, &data);

    cv::imshow(WINDOW_NAME, filter.get_result());
    cv::waitKey();

    return filter.get_params();
}
