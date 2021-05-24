// Giacomello Mattia
// I.D. 1210988

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "PanoramicImage.h"


std::vector<cv::Mat> loadImages(cv::String pattern);

/**
* To test this program are required 2 parameters:
* argv[1] is the pattern required to load the dataset, for example "..\..\data\dataset_dolomites\dolomites\\*.png"
* argv[2] is the fov associated to the dataset chosen
*/


int main(int argc, char *argv[])
{

	cv::String pattern = argv[1];
	double fov = std::atof(argv[2]);
	
	// Load of images specified by the user through the program arguments
	std::vector<cv::Mat> imageSet = loadImages(pattern);

	// Creation of the class PanoramicImage and computation of the merged image
	PanoramicImage stitcher(imageSet, fov);
	cv::Mat output = stitcher.doStitch();
	
	// Resize ot hte panoramic image for a view fitting the screen
	int widthScreen = 1920;
	cv::Mat outputResized;
	cv::resize(output, outputResized, cv::Size(widthScreen, widthScreen * output.rows / output.cols));
	
	cv::imshow("Panoramic Image", outputResized);

	cv::waitKey();

}

/**
* Method used to load the images inside a vector from a patten
* @param pattern Pattern used to specify where the images
* @return vector containing the images found
*/
std::vector<cv::Mat> loadImages(cv::String pattern)
{
	std::vector<cv::String> collection;
	std::vector<cv::Mat> imageVector;
	cv::glob(pattern, collection);	

	for (cv::String name : collection)
	{
		cv::Mat img = cv::imread(name);
		if (!img.empty()) 
			imageVector.push_back(img);
	}

	return imageVector;
}

 


