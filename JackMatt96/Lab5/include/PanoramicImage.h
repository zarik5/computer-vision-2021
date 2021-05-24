#include <opencv2/core.hpp>
#include <vector>
#include "panoramic_utils.h"


class PanoramicImage
{
public:
	/**
	* Constructor
	* @param imageVector Container of the Mat that need to be stitched from left to right.
	* @param fov Field of view used to take the images
	*/
	PanoramicImage(std::vector<cv::Mat> imageVector, double fov);
	
	/**
	* Add Mat to the dataset passed in the constructor, the new image is attached at the right 
	* @param image image to add to the vector inside the class
	*/
	void addImage(cv::Mat imageSet);
	
	/**
	* Retrieve the field of view passed in the constructor
	* @return Field of view
	*/
	double getFov();
	
	/**
	* Compute the merge of the photos stored inside the class
	* @param ratio Value that
	* @param maxRansacIter Maximum number of iteration for the RANSAC estimator
	* @param thresholdRansac Threshold for the RANSAC estimator
	* @return Panoramic photo
	*/
	cv::Mat doStitch(double ratio = 4.5, int maxRansacIter = 50, double thresholdRansac = 3);
	
	/**
	* Retrieve the panorama computed after the merge
	* @return Panoramic photo
	*/
	cv::Mat getResult();

	/**
	* Retrieve the images given to the constructor after the cylindrical projection
	* @return Dataset cylindrically projected
	*/
	std::vector<cv::Mat> getCylindricalDataset();
	

private:
	cv::Mat PanoramicImage::merge(std::vector<int> imagesPosition);
	
	std::vector<cv::Mat> dataset;
	double fov;
	cv::Mat output;
};