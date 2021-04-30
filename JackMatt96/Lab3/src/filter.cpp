#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "filter.h"

//using namespace cv;

	// constructor
	Filter::Filter(cv::Mat input_img, int size) {

		input_image = input_img;
		if (size % 2 == 0)
			size++;
		filter_size = size;
	}

	// for base class do nothing (in derived classes it performs the corresponding filter)
	void Filter::doFilter() {

		// it just returns a copy of the input image
		result_image = input_image.clone();

	}

	// get output of the filter
	cv::Mat Filter::getResult() {

		return result_image;
	}

	//set window size (it needs to be odd)
	void Filter::setSize(int size) {

		if (size % 2 == 0)
			size++;
		filter_size = size;
	}

	//get window size 
	int Filter::getSize() {

		return filter_size;
	}

	// Write your code to implement the Gaussian, median and bilateral filters

	
	//constructor
	GaussianFilter::GaussianFilter(cv::Mat input_img, int filter_size, float sigma):Filter(input_img, filter_size) {

		this->sigma = sigma;
	}

	//compute the filtered image
	void GaussianFilter::doFilter(){

		cv::GaussianBlur(input_image, result_image, cv::Size(filter_size, filter_size), sigma);
	}

	// set sigma
	void GaussianFilter::setSigma(float sigma){

		this->sigma = sigma;
	}

	// get sigma
	float GaussianFilter::getSigma(){

		return sigma;
	}



	// Contructor
	MedianFilter::MedianFilter(cv::Mat input_img, int size) :Filter(input_img, size){}

	//compute the filtered image
	void MedianFilter::doFilter(){

		cv::medianBlur(input_image, result_image, filter_size);
	}


	
	// Constructor
	BilateralFilter::BilateralFilter(cv::Mat input_img, int filter_size, float sigma_space, float sigma_range) :Filter(input_img, filter_size){

		this->sigma_range = sigma_range;
		this->sigma_space = sigma_space;
	}

	//compute the filtered image
	void BilateralFilter::doFilter(){

		cv::bilateralFilter(input_image, result_image,filter_size, sigma_range, sigma_space);
	}

	//set sigma range
	void BilateralFilter::setSigmaRange(float sigma_range){

		this->sigma_range = sigma_range;
	}

	//get sigma range
	float BilateralFilter::getSigmaRange()
	{
		return sigma_range;
	}

	//set sigma space
	void BilateralFilter::setSigmaSpace(float sigma_space){

		this->sigma_space = sigma_space;
	}

	//get sigma space
	float BilateralFilter::getSigmaSpace(){

		return sigma_space;
	}