// Giacomello Mattia
// I.D. 1210988

#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>


void findChessboardPoints(cv::String, std::vector<std::vector<cv::Vec2f>> &, std::vector<cv::Mat> &, std::vector< cv::String >&);
void calibrateCameraChessboard(std::vector<std::vector<cv::Vec2f>>& ,cv::Size , cv::Size , float ,cv::Mat& , cv::Mat& ,std::vector<std::vector<cv::Vec2f>>& );
double meanEuclidenReprojectionError(std::vector<std::vector<cv::Vec2f>>& ,std::vector<std::vector<cv::Vec2f>>& , std::vector <double>& );
double rootMeanSquaredReprojectionError(std::vector<std::vector<cv::Vec2f>>& ,std::vector<std::vector<cv::Vec2f>>& , std::vector <double>& ); 
std::vector<cv::Vec3f> chessboard3dPoints(cv::Size , float );
cv::Mat undistortImage(cv::Mat , cv::Mat , cv::Mat );


cv::Size gridCorners(7,5); // Internal corners of the chessboard
float gridMeasure = 0.03; // Chessboard edge in m 
cv::String chessboardDirectoryPath = "..\\..\\data\\"; // Directory path of the chessboard calibation images
std::string distortedImagePath = "..\\..\\data\\_DSC6070.jpg"; // Path of the image to undistort
// The self taken dataset here used can be retrieved from the link in the report

int main(){

	/**********************************
	 * First Part: Camera Calibration *
	 **********************************/

	// Extraction from the given path of chessboard points, images and image names
	std::vector<std::vector<cv::Vec2f>> chessboardImagesPoints;
    std::vector<cv::Mat> chessboardImages;
    std::vector<cv::String>imagesName;
	findChessboardPoints(chessboardDirectoryPath, chessboardImagesPoints, chessboardImages, imagesName);
	    
	// Computation of intrinsic parameters and reprojected points throught the chessboard calibration procedure
	cv::Mat cameraMatrix, distCoeffs;
	std::vector<std::vector<cv::Vec2f>> reprojectedPoints;
	calibrateCameraChessboard(chessboardImagesPoints, chessboardImages[0].size(), gridCorners, gridMeasure, cameraMatrix,  distCoeffs, reprojectedPoints);

	// Computation of mean euclidean reprojection errors and root mean squared reprojection errors
	std::vector <double> MRE, RMS;
	double totalMRE = meanEuclidenReprojectionError(chessboardImagesPoints ,reprojectedPoints, MRE);
	double totalRMS = rootMeanSquaredReprojectionError(chessboardImagesPoints ,reprojectedPoints, RMS); 
	
	int minIdx[2], maxIdx[2], minSIdx[2], maxSIdx[2];
	double minVal, maxVal, minSVal, maxSVal;
	cv::minMaxIdx(MRE, &minVal, &maxVal, minIdx, maxIdx);
	cv::minMaxIdx(RMS, &minSVal, &maxSVal, minSIdx, maxSIdx);
	

    // Print to terminal of values obtained
	std::cout << "\nCamera matrix: \n" << cameraMatrix
		<< "\n\nDistortion coefficients: \n[k1 k2 p1 p2 k3] = " << distCoeffs
		<< "\n\nTotal mean euclidean reprojection error = "<< totalMRE 
		<< "\nBest image: " << imagesName[minIdx[1]] <<" with mean reprojection error = "<< minVal 
		<< "\nWorst image: " << imagesName[maxIdx[1]] << " with mean reprojection error = " << maxVal
		<< "\n\nTotal root mean squared reprojection error = "<< totalRMS 
		<< "\nBest image: " << imagesName[minSIdx[1]] <<" with root mean squared reprojection error = "<< minSVal 
		<< "\nWorst image: " << imagesName[maxSIdx[1]] << " with root mean squared reprojection error = " << maxSVal << "\n";


	/***********************************************
	 * Second Part: Undistortion and visualization *
	 ***********************************************/
	
	// Loading and undistortion of an example image
	cv::Mat distortedImage = cv::imread(distortedImagePath);
	cv::Mat undistortedImage = undistortImage(distortedImage, cameraMatrix, distCoeffs);


	// Visualization for comparison of before and after the undistorting procedure image
	// Values to resize the images to a smaller window, in particular the maximum size will be that of a 720p 16:9 display
	int windowHeight=720, windowWidth=1280;
	cv::Size imageResizeValue;
	if(windowHeight * distortedImage.cols / distortedImage.rows>windowWidth){
		imageResizeValue = cv::Size(windowWidth,windowWidth * distortedImage.rows / distortedImage.cols);
	}
	else{
		imageResizeValue = cv::Size(windowHeight * distortedImage.cols / distortedImage.rows, windowHeight);
	}

	cv::resize(distortedImage, distortedImage, imageResizeValue,0,0,cv::INTER_LANCZOS4);	
	cv::resize(undistortedImage, undistortedImage, imageResizeValue,0,0,cv::INTER_LANCZOS4);
	cv::imshow("Original", distortedImage);
	cv::imshow("Undistorted", undistortedImage);

	cv::waitKey(); 

	return 0;
}


/**
* Method to retrieve from a directory the image's chessboard points to calibrate a camera 
* @param chessboardDirectory Path to the folder containing the images  
* @param chessboardImagesPoints Vector containing the points of the chessboard in the image plane for each image
* @param chessboardImages Vector containing the images containing a chessboard
* @param imagesName Vector containing the names of images containing a chessboard
*/
void findChessboardPoints(cv::String chessboardDirectory, std::vector<std::vector<cv::Vec2f>>& chessboardImagesPoints, std::vector<cv::Mat>& chessboardImages, std::vector< cv::String >& imagesName){
    
	// Searching for all elements in the given folder path
	std::vector< cv::String >imagesVectorPath;
    cv::glob(chessboardDirectory+"*",imagesVectorPath);

	std::cout<<"\nChessboard found in the images:\n";
	for(auto imagePath:imagesVectorPath)
    {
        cv::Mat img = cv::imread(imagePath);
		std::vector<cv::Vec2f> points;
		if(!cv::findChessboardCorners(img, gridCorners, points)){
			continue;
		}
		// Refining of chessboard corner and push in the vector
		cv::Mat gray;
		cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
		cv::cornerSubPix(gray, points, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));
		cv::drawChessboardCorners(img, gridCorners, points, true);
		chessboardImagesPoints.push_back(points);
		chessboardImages.push_back(img);
        imagesName.push_back(imagePath.substr(imagePath.find_last_of("\\")+1));
		std::cout<<imagesName.back()<<"\n";
    }
	std::cout<<chessboardImages.size()<<" image containing the chessboard calibration pattern found\n";
}


/**
* Method to compute the intrinsic parameters of the camera from a chessboard pattern
* @param chessboardImagesPoints Vector containing the points of the chessboard in the image plane for each image
* @param imageSize Size of the images
* @param gridSize Size element containing the number of edges per row and column
* @param gridMeasure Measure in m of the edge of the chessboard squares
* @param cameraMatrix Camera matrix
* @param distCoeffs Vector containing the distortion coefficients [k1 k2 p1 p2 k3]
* @param projectedPoints Vector containing the points of the chessboard projected from 3D world to image plane through the calibration parameters
*/
void calibrateCameraChessboard(std::vector<std::vector<cv::Vec2f>>& chessboardImagesPoints,cv::Size imageSize, cv::Size gridSize, float gridMeasure,cv::Mat &cameraMatrix, cv::Mat &distCoeffs,std::vector<std::vector<cv::Vec2f>>& projectedPoints){  
	
	// Computation of the chessboard pattern in vector form	
	std::vector<cv::Vec3f> chessboard = chessboard3dPoints(gridSize,gridMeasure);

	// Computation of camera intrinsic and extrinsic parameters and distiortion coefficients	
	std::vector <cv::Mat> rvecs, tvecs;
	std::cout<<cv::calibrateCamera(std::vector<std::vector<cv::Vec3f>> (chessboardImagesPoints.size(),chessboard), chessboardImagesPoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs)<<"\n";

	projectedPoints = std::vector<std::vector<cv::Vec2f>> (chessboardImagesPoints.size());
	for (size_t i = 0; i < chessboardImagesPoints.size(); i++)
	{
		cv::projectPoints(chessboard, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projectedPoints[i]);
	}
}

/**
* Method to compute the mean reprojection errors of a reprojected chessboard
* @param chessboardImagesPoints Vector containing the points of the chessboard in the image plane for each image
* @param projectedPoints Vector containing the points of the chessboard projected from 3D world to image plane through the calibration parameters
* @param MRE Output vector of mean reprojection error per picture 
* @return Total mean reprojection error 
*/
double meanEuclidenReprojectionError(std::vector<std::vector<cv::Vec2f>>& chessboardImagesPoints,std::vector<std::vector<cv::Vec2f>>& projectedPoints, std::vector <double>& MRE){  
	
    // Computation of mean euclidean reprojection errors
	MRE = std::vector <double> (chessboardImagesPoints.size());
    double totalError = 0;
	for (size_t i = 0; i < chessboardImagesPoints.size(); i++)
	{
		double error = 0;
		for (size_t j = 0; j < chessboardImagesPoints[i].size(); j++)
		{
			error += cv::norm(chessboardImagesPoints[i][j]- projectedPoints[i][j]);
		}
        MRE[i] = error / chessboardImagesPoints[i].size();
        totalError += error;      
	}
	return totalError/(chessboardImagesPoints[0].size()*chessboardImagesPoints.size());
}

/**
* Method to compute the root mean squared reprojection errors of a reprojected chessboard
* @param chessboardImagesPoints Vector containing the points of the chessboard in the image plane for each image
* @param projectedPoints Vector containing the points of the chessboard projected from 3D world to image plane through the calibration parameters
* @param RMS Output vector of mean squared reprojection errors per picture 
* @return Total mean squared reprojection errors 
*/
double rootMeanSquaredReprojectionError(std::vector<std::vector<cv::Vec2f>>& chessboardImagesPoints,std::vector<std::vector<cv::Vec2f>>& projectedPoints, std::vector <double>& RMS ){  
	
    // Computation of root mean squared reprojection errors
	RMS = std::vector <double> (chessboardImagesPoints.size());
    double totalError = 0;
	for (size_t i = 0; i < chessboardImagesPoints.size(); i++)
	{
		double error = 0;
		for (size_t j = 0; j < chessboardImagesPoints[i].size(); j++)
		{
			error += cv::pow(cv::norm(chessboardImagesPoints[i][j]- projectedPoints[i][j]),2);
		}
        RMS[i] = cv::sqrt(error / chessboardImagesPoints[i].size());
		totalError += error;      
	}      
	return sqrt(totalError/(chessboardImagesPoints[0].size()*chessboardImagesPoints.size()));
}

/**
* Method to build the 3D reference chessboard with Z=0. The top left corner is placed to (0,0,0)
* @param gridSize Size element containing the number of edges per row and column
* @param gridMeasure Measure in m of the edge of the chessboard squares
* @return vector containing the 3D world points of the chessboard corners
*/
std::vector<cv::Vec3f> chessboard3dPoints(cv::Size gridSize, float gridMeasure){
	// Creation of vector of virtual chessboard
	std::vector<cv::Vec3f> chessboard;
	for (size_t i = 0; i < gridSize.height; i++)
	{
		for (size_t j = 0; j < gridSize.width; j++)
		{
			chessboard.push_back(cv::Vec3f(j * gridMeasure, i * gridMeasure, 0));
		}
	}
	return chessboard;
}


/**
* Method to undistort an image
* @param distortedImage Distorted image
* @param cameraMatrix Camera matrix
* @param distCoeffs Vector containing the distortion coefficients [k1 k2 p1 p2 k3]
* @return Undistorted image
*/
cv::Mat undistortImage(cv::Mat distortedImage, cv::Mat cameraMatrix, cv::Mat distCoeffs){
	
	// Computation of mapping matrices 
	cv::Mat mapx, mapy;
	cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix, distortedImage.size(), CV_32FC1, mapx, mapy);
	
	// Remapping of distorted image
	cv::Mat undistortedImage;
	cv::remap(distortedImage, undistortedImage, mapx,mapy, cv::INTER_LANCZOS4);
	return undistortedImage;
}