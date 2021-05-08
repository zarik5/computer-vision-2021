#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Generic class implementing a filter with the input and output image data and the parameters
class Filter{

// Methods

public:

	// constructor 
	// input_img: image to be filtered
	// filter_size : size of the kernel/window of the filter
	Filter(cv::Mat input_img, int filter_size);

	// perform filtering (in base class do nothing, to be reimplemented in the derived filters)
	void doFilter();

	// get the output of the filter
	cv::Mat getResult();

	//set the window size (square window of dimensions size x size)
	void setSize(int size);
	
	//get the Window Size
	int getSize();

// Data

protected:

	// input image
	cv::Mat input_image;

	// output image (filter result)
	cv::Mat result_image;

	// window size
	int filter_size;

};



// Gaussian Filter
class GaussianFilter : public Filter  {

public:
	// constructor 
	// input_img: image to be filtered
	// filter_size : size of the kernel/window of the filter
	// sigma : standard deviation of gaussian kernel
	GaussianFilter(cv::Mat input_img, int filter_size, float sigma);

	// Compute the filtered image with gaussian filter
	void doFilter();

	// Set the sigma
	// sigma : standard deviation of gaussian kernel
	void setSigma(float sigma);

	// Return the sigma
	float getSigma();

private:
	// standard deviation of gaussian kernel
	float sigma;

};



// Median Filter
class MedianFilter : public Filter {

public:
	// constructor 
	// input_img: image to be filtered
	// filter_size : size of the kernel/window of the filter
	MedianFilter(cv::Mat input_img, int filter_size);
	// Compute the filtered image with gaussian filter
	void doFilter();

};



// Bilateral Filter
class BilateralFilter : public Filter {

public:
	// constructor 
	// input_img: image to be filtered
	// filter_size : size of the kernel/window of the filter
	// sigma_range : standard deviation of color range
	// sigma_space : standard deviation of space
	BilateralFilter(cv::Mat input_img, int filter_size, float sigma_space, float sigma_range);

	// Compute the filtered image with gaussian filter
	void doFilter();

	// Set the sigma range
	// sigma_range : standard deviation of color range
	void setSigmaRange(float sigma_range);

	// Return the sigma range
	float getSigmaRange();

	// Set the sigma space
	// sigma_space : standard deviation of space
	void setSigmaSpace(float sigma_space);

	// Return the sigma space
	float getSigmaSpace();

private:
	// standard deviation of color range
	float sigma_range;
	// standard deviation of space
	float sigma_space;

};