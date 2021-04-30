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
