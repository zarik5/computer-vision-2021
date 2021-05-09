#include "filter.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

static int make_odd(int value) { return (value / 2) * 2 + 1; }

// Filter methods
Filter::Filter(cv::Mat input_img) : input_image(input_img) {}
cv::Mat Filter::get_result() { return result_image; }

// MedianFilter methods
MedianFilter::MedianFilter(cv::Mat input_img, MedianFilterParams initial_params)
    : Filter(input_img), params(initial_params) {
    // Run filter once to make sure get_result() always returns a valid image
    do_filter();
}
void MedianFilter::do_filter() {
    this->params.kernel_size = make_odd(this->params.kernel_size);
    cv::medianBlur(this->input_image, this->result_image, this->params.kernel_size);
}
MedianFilterParams MedianFilter::get_params() { return this->params; }
int *MedianFilter::kernel_size_ptr_mut() { return &this->params.kernel_size; }

// GaussianFilter methods
GaussianFilter::GaussianFilter(cv::Mat input_img, GaussianFilterParams initial_params)
    : Filter(input_img), params(initial_params) {
    do_filter();
}
void GaussianFilter::do_filter() {
    this->params.kernel_size = make_odd(this->params.kernel_size);
    cv::GaussianBlur(this->input_image, this->result_image,
                     {this->params.kernel_size, this->params.kernel_size}, this->params.sigma);
}
GaussianFilterParams GaussianFilter::get_params() { return this->params; }
int *GaussianFilter::kernel_size_ptr_mut() { return &this->params.kernel_size; }
int *GaussianFilter::sigma_ptr_mut() { return &this->params.sigma; }

// BilateralFilter methods
BilateralFilter::BilateralFilter(cv::Mat input_img, BilateralFilterParams initial_params)
    : Filter(input_img), params(initial_params) {
    do_filter();
}
void BilateralFilter::do_filter() {
    this->params.kernel_size = make_odd(6 * this->params.sigma_space);

    cv::bilateralFilter(this->input_image, this->result_image, this->params.kernel_size,
                        this->params.sigma_range, this->params.sigma_space);
}
BilateralFilterParams BilateralFilter::get_params() { return this->params; }
int *BilateralFilter::sigma_range_ptr_mut() { return &this->params.sigma_range; }
int *BilateralFilter::sigma_space_ptr_mut() { return &this->params.sigma_space; }