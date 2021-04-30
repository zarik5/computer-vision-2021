#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <stdlib.h>
#include <iostream>
#include <opencv2/video/tracking.hpp>

class VideoObjectTracking
{
public:
	// Constructor: Load video frames and objects to track
	// @param pathVideo: path of video
	// @param pathObjectsFolder folder containing images of objects to track
	// @param videoOutputName: name of the video with traked objects
	VideoObjectTracking(std::string pathVideo, std::string pathObjectsFolder);

	// Find features and boundaries of object to track
	// @param ratio: ratio between the minimum and the maximum accepted distance between descriptors 
	// @return first frame with found object boundaries
	cv::Mat matchFeatures(float ratio=5);

	// Create the output video with tracked objects and direction of keypoints motion
	// @param ratio: ratio between the minimum and the maximum accepted distance between descriptors 
	// @return tracked video
	std::vector<cv::Mat> computeTrackedFrames(float ratio = 5);

	// Returns the frames with tracking
	// @return tracked video
	std::vector<cv::Mat> getResults();

	// Returns the original frames
	// @return vector of frames of original video
	std::vector<cv::Mat> getOriginalFrames();

	// Return the first frame with bound for each object
	// @return first frame with boundaries
	cv::Mat getMatchFeatures();

private:
	std::vector<cv::Mat> framesVector;
	std::vector<cv::Mat> ResultVector;
	std::vector<cv::Mat> objImages;
	std::vector<std::vector<cv::Point2f>> GoodPoints;
	std::vector<std::vector<cv::Point2f>> detectedCorners;
	cv::Mat frameStart;

};
