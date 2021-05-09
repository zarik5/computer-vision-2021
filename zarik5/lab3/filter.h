#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Generic class implementing a filter with the input and output image data and the parameters
class Filter {

    // Methods

  public:
    // constructor
    // input_img: image to be filtered
    // initial_parameters: initial value for parameters
    Filter(cv::Mat input_img);

    // perform filtering
    virtual void do_filter() = 0;

    // get the output of the filter
    cv::Mat get_result();

    // Data

  protected:
    // input image
    cv::Mat input_image;

    // output image (filter result)
    cv::Mat result_image;
};

struct MedianFilterParams {
    int kernel_size;
};

class MedianFilter : public Filter {
  public:
    MedianFilter(cv::Mat input_img, MedianFilterParams initial_params);
    virtual void do_filter() override;
    MedianFilterParams get_params();

    // Get a mutable pointer to the kernel_size parameter
    int *kernel_size_ptr_mut();

  private:
    MedianFilterParams params;
};

struct GaussianFilterParams {
    int kernel_size;
    int sigma;
};

class GaussianFilter : public Filter {
  public:
    GaussianFilter(cv::Mat input_img, GaussianFilterParams initial_params);
    virtual void do_filter() override;
    GaussianFilterParams get_params();
    int *kernel_size_ptr_mut();
    int *sigma_ptr_mut();

  private:
    GaussianFilterParams params;
};

struct BilateralFilterParams {
    int kernel_size;
    int sigma_range;
    int sigma_space;
};

class BilateralFilter : public Filter {
  public:
    BilateralFilter(cv::Mat input_img, BilateralFilterParams initial_params);
    virtual void do_filter() override;
    BilateralFilterParams get_params();
    int *sigma_range_ptr_mut();
    int *sigma_space_ptr_mut();

  private:
    BilateralFilterParams params;
};