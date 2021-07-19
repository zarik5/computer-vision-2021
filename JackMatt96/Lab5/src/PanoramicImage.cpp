// Giacomello Mattia
// I.D. 1210988

#include "PanoramicImage.h"


PanoramicImage::PanoramicImage(std::vector<cv::Mat> imageVector, double fov)
{
	this->fov = fov;
	
	dataset.reserve(imageVector.size());

	double angle = fov/2;

	for(auto image: imageVector)
	{
		dataset.push_back(PanoramicUtils::cylindricalProj(image,angle));
	}

	output = cv::Mat();
}



void PanoramicImage::addImage(cv::Mat image)
{
	dataset.push_back(PanoramicUtils::cylindricalProj(image,fov/2));
}



double PanoramicImage::getFov()
{
	return fov;
}



cv::Mat PanoramicImage::doStitch(double ratio)
{
	// Cration of the SIFT feature extractor and brute force matcher with l2 norm
	cv::Ptr <cv::SIFT> sift = cv::SIFT::create(5000);
	cv::Ptr <cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2);

	// Computation of all the keypoints and the relative descriptors
	std::vector <std::vector<cv::KeyPoint>> keypoints;
	std::vector<cv::Mat> descriptors;
	sift->detect(dataset, keypoints);
	sift->compute(dataset, keypoints, descriptors);

	std::vector<int> imagesPosition(dataset.size());	
	// Setting the position of the first image as the left margin image
	imagesPosition[0] = 0;

	for (int i = 1; i < dataset.size(); i++)
	{
		// Computation of the brute force matches between two consecutive images
		std::vector<cv::DMatch> matchesFound;
		matcher->match(descriptors[i],descriptors[i-1],matchesFound);

		// Matches refining using an upper bound in the distance computed as minimum distance times a constant
		double min = matchesFound[0].distance;
		for (int j = 0; j < matchesFound.size(); j++)
		{
			if(matchesFound[j].distance<min)
				min = matchesFound[j].distance;
		}
		double upperBound = min * ratio;
		
		// Extraction of the points that have low distances
		std::vector<cv::Point2f> leftPoints, rightPoints;
		for(int j=1; j < matchesFound.size();j++)
		{
			if(matchesFound[j].distance<upperBound)
			{
				// Computation of the distance between the positions of the matching points in the images
				rightPoints.push_back(keypoints[i - 1][matchesFound[j].trainIdx].pt);
				leftPoints.push_back(keypoints[i][matchesFound[j].queryIdx].pt);
			}
		}
		// Computation of translation using the inliners given by the RANSAC procedure done inside the findHomography method
		std::vector<uchar> goodMatches;
		cv::findHomography(rightPoints, leftPoints, goodMatches ,cv::RANSAC);

		double sumOfDelta = 0; 
		int counter = 0;
		for(int j = 0; j<rightPoints.size();j++)
		{
			if(goodMatches[j])
			{
				sumOfDelta += rightPoints[j].x-leftPoints[j].x;
				counter++;
			}
		}
		// Computation of the position of the image
		imagesPosition[i] = imagesPosition[i-1] + int(sumOfDelta/counter);		
	}

	// Computation of the output size of the output image and stitch of the images with the correct translation 
	output = cv::Mat(dataset.back().rows, dataset.back().cols + imagesPosition.back(), dataset.back().type());
	for (int i = 0; i < dataset.size(); i++)
	{
		// To obtain a better result before the stitching is done a histogram equalization 
		cv::Mat temp;
		cv::equalizeHist(dataset[i],temp);
		cv::Mat areaToCopy = output.colRange(imagesPosition[i], imagesPosition[i]+dataset[i].cols);
		temp.copyTo(areaToCopy);
	}

	return output;
}



cv::Mat PanoramicImage::getResult()
{
	return output;
}



std::vector<cv::Mat> PanoramicImage::getCylindricalDataset()
{
	return dataset;
}