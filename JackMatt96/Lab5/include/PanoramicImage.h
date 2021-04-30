#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "panoramic_utils.h"

using namespace cv;
using namespace std;

class PanoramicImage
{
public:
	/**
	* Constructor
	* @param imageSet Container of the Mat that need to be stitched from left to right.
	* @param fov Field of view
	*/
	PanoramicImage(vector<Mat> imageSet, float fov);
	
	/**
	* Add Mat to the dataset passed in the constructor, the new images are attached at the right 
	* @param imageSet Container of Mat
	*/
	void addImages(vector<Mat> imageSet);
	
	/**
	* Retrieve the field of view passed in the constructor
	* @return Field of view
	*/
	float getFov();
	
	/**
	* Compute the merge of the photos
	* @param ratio Value that
	* @param orbPoints Keypoints required at the ORB feature detector
	* @param maxRansacIter Maximum number of iteration for the RANSAC estimator
	* @param thresholdRansac Threshold for the RANSAC estimator
	*/
	void doStitch(float ratio = 3, int orbPoints = 5000, int maxRansacIter = 50, int thresholdRansac = 3);
	
	/**
	* Retrieve the panorama computed after the merge
	* @return Panoramic photo
	*/
	Mat getResult();

	/**
	* Retrieve the images given to the constructor after the cylindrical projection
	* @return Dataset cylindrically projected
	*/
	vector<Mat> getDataset();


private:
	vector<Mat> dataset;
	float angle;
	Mat output;
};