
#include "VideoObjectTracking.h"



VideoObjectTracking::VideoObjectTracking(std::string pathVideo, std::string pathObjectsFolder)
{
	// Open a video stream and load single frames in a vector
	cv::VideoCapture capture(pathVideo);
	
	if (capture.isOpened())
		{
		while (true)
			{
			cv::Mat frame;
			capture >> frame;
			if (frame.empty())
				break;
			framesVector.push_back(frame);

			}
		}

	// Load the object to track in a vector
	std::vector<std::string> objNames;
	cv::glob(pathObjectsFolder, objNames);
	for (std::string name : objNames)
	{
		cv::Mat img = cv::imread(name);
		objImages.push_back(img);

	}

}



cv::Mat VideoObjectTracking::matchFeatures(float ratio){

	std::vector<std::vector<cv::KeyPoint>> keypointsObj;
	std::vector<cv::Mat> descriptorsObj;
	std::vector<cv::KeyPoint> keypoints(objImages.size());
	cv::Mat descriptors;
	frameStart = framesVector[0];

	GoodPoints = std::vector<std::vector<cv::Point2f>>();
	detectedCorners = std::vector<std::vector<cv::Point2f>>();

	// Cration of the SIFT descriptor and of the brute force matcher with L2 norm
	// Computation of all the keypoints and the relative descriptors

	cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
	siftPtr->detect(objImages, keypointsObj);
	siftPtr->compute(objImages, keypointsObj, descriptorsObj);
	siftPtr->detect(framesVector[0], keypoints);
	siftPtr->compute(framesVector[0], keypoints, descriptors);

	cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create();

	// Find matches between objects and image keypoints and draw found object boundaries
	for (size_t i = 0; i < objImages.size(); i++)
	{
		std::vector<cv::DMatch> matches;
		matcher->match(descriptors, descriptorsObj[i], matches);
		sort(matches.begin(), matches.end());
		float upper = matches[0].distance * ratio;
		size_t j = 0;

		std::vector<cv::Point2f> pointsObj;
		std::vector<cv::Point2f> pointsImg;

		// Refinement of matches using the formula ratio*minDistance as maximum distance accepted
		while ((j < matches.size()) && (matches[j].distance <= upper))
		{
			pointsObj.push_back(keypointsObj[i][matches[j].trainIdx].pt);
			pointsImg.push_back(keypoints[matches[j].queryIdx].pt);
			j++;
		}

		// Find the perspective transformation between objects and frame and use it to project the corners 

		cv::Mat mask;
		cv::Mat H = cv::findHomography(pointsObj, pointsImg,cv::RANSAC,3,mask);

		std::vector<cv::Point2f> cornersObj;
		cornersObj = { cv::Point2f(0,0),
			cv::Point2f(0,objImages[i].rows-1),
			cv::Point2f(objImages[i].cols - 1,objImages[i].rows - 1),
			cv::Point2f(objImages[i].cols-1,0)
		};

		std::vector<cv::Point2f> cornersImg(4);
		cv::perspectiveTransform(cornersObj, cornersImg, H);


		// Draw the rectangular border of objects
		for (size_t j = 0; j < 4; j++)
		{
			cv::line(frameStart, cornersImg[j], cornersImg[(j + 1) % 4], cv::Scalar(0, 0, 255));
		}

		// Save corners and keypoints filtered through Ransac inliners
		std::vector<cv::Point2f> tmp;
		for (size_t j = 0; j < mask.rows; j++)
		{
			if (mask.at<int>(j,0)==1)
				tmp.push_back(pointsImg[j]);
		}
		GoodPoints.push_back(tmp);
		detectedCorners.push_back(cornersImg);
	}

	return frameStart;

}


std::vector<cv::Mat> VideoObjectTracking::computeTrackedFrames(float ratio){

	matchFeatures(ratio);

	ResultVector = std::vector<cv::Mat>();
	ResultVector.push_back(frameStart);

	// For each pair of consecutive frames calculate the optical flow of each object with Lukas-Kanade method.
	for (size_t i = 0; i < framesVector.size() - 1; i++) {
			
			cv::Mat newFrame = framesVector[i+1].clone();
			for (size_t j = 0; j < GoodPoints.size(); j++) {

				std::vector<cv::Point2f> NewGoodPoints;
				std::vector<unsigned char> status;
				std::vector<float> err;

				// Compute optical flow with Lukas-Kanade method
				cv::calcOpticalFlowPyrLK(framesVector[i], framesVector[i + 1], GoodPoints[j], NewGoodPoints, status, err,cv::Size(11,11));
				
				// Find perspective tranformation between keypoints and compute reprojection
				cv::Mat H = cv::findHomography(GoodPoints[j], NewGoodPoints,cv::RANSAC);
				std::vector<cv::Point2f> cornersImg(4);
				cv::perspectiveTransform(detectedCorners[j], cornersImg, H);
				detectedCorners[j] = cornersImg;

				// Draw object margins and motion direction of keypoints
				for (size_t k = 0; k < 4; k++) {
					cv::line(newFrame, cornersImg[k],
							cornersImg[(k + 1) % 4], cv::Scalar(0, 0, 255));

				}

				for(size_t k=0, n=0; k<GoodPoints[j].size();k++){

					if(status[k]==1){
						cv::arrowedLine(newFrame, GoodPoints[j][k], NewGoodPoints[n], cv::Scalar(0,255,255),3);
						n++;
					}

				}

				GoodPoints[j]= NewGoodPoints;
			}
			
			ResultVector.push_back(newFrame);
		}
	return ResultVector;
}

std::vector<cv::Mat> VideoObjectTracking::getResults()
{
	return ResultVector;
}

std::vector<cv::Mat> VideoObjectTracking::getOriginalFrames()
{
	return framesVector;
}

cv::Mat VideoObjectTracking::getMatchFeatures()
{
	return frameStart;
}

