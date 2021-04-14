// Giacomello Mattia
// I.D. 1210988


#include "PanoramicImage.h"


PanoramicImage::PanoramicImage(vector<Mat> imageSet, float fov)
{
	angle = fov / 2;
	for (Mat img : imageSet)
	{
		dataset.push_back(PanoramicUtils::cylindricalProj(img, angle));
	}
}

void PanoramicImage::addImages(vector<Mat> imageSet)
{
	for (Mat img : imageSet)
	{
		dataset.push_back(PanoramicUtils::cylindricalProj(img, angle));
	}
}

float PanoramicImage::getFov()
{
	return angle*2;
}

void PanoramicImage::doStitch(float ratio, int orbPoints, int maxRansacIter, int thresholdRansac)
{
	vector < vector<KeyPoint>> keypoints;
	vector<Mat> descriptors;
	
	// Cration of the ORB descriptor and of the brute force matcher with hamming norm. For better result is activated the cross check
	Ptr <ORB> orb = ORB::create(orbPoints);
	Ptr <BFMatcher> matcher = BFMatcher::create(NORM_HAMMING, true);

	// Computation of all the keypoints and the relative descriptors
	orb->detect(dataset, keypoints);
	orb->compute(dataset, keypoints, descriptors);

	// Setting the translation of the first image
	vector<Point2d> translations(dataset.size());
	translations[0] = Point2d(0, 0);

	int ymin = 0, ymax = 0;

	for (size_t i = 1; i < keypoints.size(); i++)
	{
		// Match of the orb feature found between two consective images
		vector<DMatch> matches;
		matcher->match(descriptors[i], descriptors[i - 1], matches);

		// Sorting of the matches by the hamming distance selection of the best by an upper limit
		sort(matches.begin(), matches.end());
		float upper = matches[0].distance * ratio;
		size_t j = 0;

		vector<Point2d> pointsDxy;

		while ((j < matches.size()) && (matches[j].distance <= upper))
		{
			// Computation of the distance between the positions of the matching points in the images
			Point2d point = keypoints[i - 1][matches[j].trainIdx].pt - keypoints[i][matches[j].queryIdx].pt;
			pointsDxy.push_back(point);
			j++;
		}


		// RANSAC made to retrieve the translation between the keypoints.
		Point2d best;
		int bestCount = 0;
		for (size_t iter = 0; iter < maxRansacIter; iter++)
		{
			size_t index = rand() % pointsDxy.size();
			int count = 0;
			for (Point2d point : pointsDxy)
			{
				// Counting of the inliers
				if (abs(point.x - pointsDxy[index].x) + abs(point.y - pointsDxy[index].y) < thresholdRansac)
					count++;
			}

			// Updating the best model in case of higher number of inliers
			if (count > bestCount)
			{
				best = pointsDxy[index];
				bestCount = count;
			}
		}

		// Computation of the translation
		Point2d d(0, 0);
		for (Point2d point : pointsDxy)
		{
			if (abs(point.x - best.x) + abs(point.y - best.y) < thresholdRansac)
			{
				d += point;
			}
		}

		translations[i] = translations[i - 1] + d / bestCount;
		
		// Since moving vertically the images creates black areas the minimum and maximum translations are needed afterwards
		if (translations[i].y < ymin)
			ymin = translations[i].y;
		else
			if (translations[i].y > ymax)
				ymax = translations[i].y;
	}

	// Creation of a temporary image with the black areas. The minimum vertical translation is used as bias to avoid negative index
	Mat out(dataset.back().rows+(ymax-ymin), dataset.back().cols + translations.back().x, dataset.back().type());

	// Merge of the images by the translations obtained previously
	for (size_t i = 0; i < dataset.size(); i++)
	{
		dataset[i].copyTo(out(Rect(translations[i].x, translations[i].y - ymin, dataset[i].cols, dataset[i].rows)));
	}

	// Computing the resulting image removing the black spots. The upper limits is the maximum translation unbiased and the height of the image is the height of a single image minus the maximum translation
	out(Rect(0, ymax - ymin, out.cols, dataset.back().rows - ymax)).copyTo(output);

}

Mat PanoramicImage::getResult()
{
	return output;
}

vector<Mat> PanoramicImage::getDataset()
{
	return dataset;
}



