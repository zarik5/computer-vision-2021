#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdlib.h>
#include <iostream>
#include "VideoObjectTracking.h"

// Video and objects paths. 
const std::string videoPath = "./Lab 6 data/video.mov";
const std::string objPath = "./Lab 6 data/objects/*.png";

int main(){

	// Creates the object tracker for the input video 
	VideoObjectTracking tracker = VideoObjectTracking(videoPath, objPath);
	
	// Finds and shows the matches in the first frame
	cv::Mat imgMatches = tracker.matchFeatures();	
	cv::imshow("Matches", imgMatches);
	cv::waitKey(0);

	// Computes and shows the tracking in the video frames
	std::vector<cv::Mat> results = tracker.computeTrackedFrames();

	for (int i = 0; i < results.size(); i++) {
		cv::imshow("Video", results[i]);
		cv::waitKey(33); // Wait to obtain a 30 fps video
	}

	return 0;
}
